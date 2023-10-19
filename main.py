import os
import torch
from transformers import AutoTokenizer, AutoProcessor, AutoModelForCausalLM, AutoModel, TrainerCallback
import nltk
import evaluate
import numpy as np
from transformers import Seq2SeqTrainer, Seq2SeqTrainingArguments
from dataset.cap_data import ImageDataset
from PIL import Image
import argparse
from transformers.trainer_callback import ProgressCallback
from models import RLVisionEncoderDecoderModel
from transformers.modeling_outputs import BaseModelOutputWithPooling


def on_log(self, args, state, control, logs=None, **kwargs):
    if state.is_local_process_zero and self.training_bar is not None:
        _ = logs.pop("total_flos", None)


def tokenization_fn(captions, max_target_length=120):
    """Run tokenization on captions."""
    labels = tokenizer(
        captions,
        padding="max_length",
        max_length=max_target_length,
        return_tensors="pt",
        truncation=True).input_ids

    return labels


def feature_extraction_fn(image_paths):
    images = [Image.open(image_file).convert('RGB') for image_file in image_paths]

    encoder_inputs = feature_extractor(images=images, return_tensors="pt")

    return encoder_inputs.pixel_values


def postprocess_text(preds, labels):
    preds = [pred.strip() for pred in preds]
    labels = [label.strip() for label in labels]
    # rougeLSum expects newline after each sentence
    preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
    labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
    return preds, labels


def collate_fn(batch):
    model_inputs = {'labels': [], 'encoder_outputs': []}
    encoder_outputs = {'pooler_output': [], 'last_hidden_state': []}
    for obj in batch:
        model_inputs['labels'].append(obj[1])
        encoder_outputs['pooler_output'].append(obj[0]['pooler_output'][0])
        encoder_outputs['last_hidden_state'].append(obj[0]['last_hidden_state'][0])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    model_inputs['encoder_outputs'] = BaseModelOutputWithPooling(
        last_hidden_state=torch.stack(encoder_outputs['last_hidden_state']),
        pooler_output=torch.stack(encoder_outputs['pooler_output']),
    )
    return model_inputs


def collate_fn_rl(batch):
    model_inputs = {'labels': [], 'pixel_values': []}
    for obj in batch:
        model_inputs['labels'].append(obj[1])
        model_inputs['pixel_values'].append(obj[0])
    model_inputs['labels'] = tokenization_fn(model_inputs['labels'])
    model_inputs['pixel_values'] = feature_extraction_fn(model_inputs['pixel_values'])
    return model_inputs


def compute_metrics(eval_preds):
    preds, labels = eval_preds
    if isinstance(preds, tuple):
        preds = preds[0]
    decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
    if ignore_pad_token_for_loss:
        # Replace -100 in the labels as we can't decode them.
        labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    # Some simple post-processing
    decoded_preds, decoded_labels = postprocess_text(decoded_preds,
                                                     decoded_labels)
    rouge_result = rouge.compute(predictions=decoded_preds,
                                 references=decoded_labels,
                                 use_stemmer=True)
    result = {k: round(v * 100, 4) for k, v in rouge_result.items()}
    # bleu_result = bleu.compute(predictions=decoded_preds,
    #                            references=decoded_labels)
    # result.update({k: round(v * 100, 4) for k, v in bleu_result.items()})
    prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
    result["gen_len"] = np.mean(prediction_lens)
    return result


class EvaluateFirstStepCallback(TrainerCallback):
    def on_step_begin(self, args, state, control, **kwargs):
        if state.global_step == 25003:
            control.should_evaluate = True


ProgressCallback.on_log = on_log
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('expname', type=str)
    parser.add_argument('--bs', type=int, default=64)
    parser.add_argument('--epochs', type=int, default=12)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--rl', action='store_true')
    parser.add_argument('--logdir', type=str, default='./logs')
    args = parser.parse_args()
    expname = args.expname + f'_{args.bs}'
    logdir = os.path.join(args.logdir, expname)
    print(expname, flush=True)
    if args.rl:
        args.resume = False
        args.bs = 1

    if os.path.exists("/project/lt200060-capgen/palm/"):
        vit_model = "/project/lt200060-capgen/palm/huggingface/vit-base-patch16-224-in21k"
        text_decode_model = args.decoder
        src_dir = "/project/lt200060-capgen/palm/capocr/data6"
        val_json = '/project/lt200060-capgen/palm/capocr/data6/val.jsonl'
        train_json = '/project/lt200060-capgen/palm/capocr/data6/train.jsonl'
        config_file = '/home/nhongcha/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = '/project/lt200060-capgen/palm/pretrained/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        output_dir = os.path.join('/project/lt200060-capgen/palm/rl-caption/workdir/', expname)
        bleu_path = '/home/nhongcha/hf-caption/bleu/bleu.py'
        rouge_path = '/home/nhongcha/hf-caption/rouge/'
        bs = args.bs
        workers = 4
        disable_tqdm = True
    elif os.path.exists("/media/palm/Data/capgen/"):
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/media/palm/Data/ocr/"
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = ''
        output_dir = os.path.join('/tmp/out/mm_dino_8x8')
        bleu_path = 'bleu'
        rouge_path = 'rouge'
        bs = 1
        workers = 0
        disable_tqdm = False
    else:
        vit_model = "google/vit-base-patch16-224-in21k"
        text_decode_model = "gpt2"
        src_dir = "/home/palm/data/coco/images"
        feats_dir = "/home/palm/data/coco/features"
        train_json = '/home/palm/data/coco/annotations/annotations/captions_train2017.json'
        val_json = '/home/palm/data/coco/annotations/annotations/captions_val2017.json'
        config_file = '/home/palm/PycharmProjects/mmdetection/configs/dino/dino-4scale_r50_8xb2-12e_coco.py'
        detector_weight = '/home/palm/PycharmProjects/mmdetection/cp/dino-4scale_r50_8xb2-12e_coco_20221202_182705-55b2bba2.pth'
        output_dir = os.path.join('workdir', expname)
        bleu_path = 'bleu'
        rouge_path = 'rouge'
        bs = 8
        workers = 0
        disable_tqdm = False
    rouge = evaluate.load(rouge_path)
    bleu = evaluate.load(bleu_path)
    os.makedirs(os.path.join(output_dir, 'train'), exist_ok=args.overwrite or args.resume)
    os.makedirs(logdir, exist_ok=args.overwrite or args.resume)
    ignore_pad_token_for_loss = True
    encoder = AutoModel.from_pretrained(vit_model)
    decoder = AutoModelForCausalLM.from_pretrained(text_decode_model, add_cross_attention=True)
    model = RLVisionEncoderDecoderModel(args.rl, None, encoder, decoder)
    model.load_state_dict(torch.load('workdir/train/pytorch_model.bin'), strict=False)
    feature_extractor = AutoProcessor.from_pretrained("openai/clip-vit-base-patch32")

    tokenizer = AutoTokenizer.from_pretrained(text_decode_model)
    tokenizer.pad_token = tokenizer.eos_token

    # update the model config
    model.config.eos_token_id = tokenizer.eos_token_id
    model.config.decoder_start_token_id = tokenizer.bos_token_id
    model.config.pad_token_id = tokenizer.pad_token_id
    if not args.resume:
        model.save_pretrained(os.path.join(output_dir, 'train'))
        feature_extractor.save_pretrained(os.path.join(output_dir, 'train'))
        tokenizer.save_pretrained(os.path.join(output_dir, 'train'))
    dir = feats_dir if not args.rl else src_dir
    train_set = ImageDataset(
        train_json,
        dir,
        args.rl,
        is_training=True,
    )
    print(len(train_set), flush=True)
    valid_set = ImageDataset(
        val_json,
        dir,
        args.rl,
        is_training=False,
    )
    print(len(valid_set), flush=True)
    # train_loader = DataLoader(train_set, **train_hyperparams)
    # valid_loader = DataLoader(valid_set, **valid_hyperparams)

    training_args = Seq2SeqTrainingArguments(
        predict_with_generate=True,
        evaluation_strategy="epoch",
        save_strategy="steps",
        save_steps=5000,
        save_total_limit=1,
        per_device_train_batch_size=bs,
        per_device_eval_batch_size=bs,
        num_train_epochs=args.epochs,
        output_dir=os.path.join(output_dir, 'train'),
        logging_dir=logdir,
        dataloader_num_workers=workers,
        logging_strategy='steps',
        logging_steps=100,
        disable_tqdm=disable_tqdm,
        report_to=['tensorboard']
    )
    trainer = Seq2SeqTrainer(
        model=model,
        # tokenizer=feature_extractor,
        args=training_args,
        compute_metrics=compute_metrics,
        train_dataset=train_set,
        eval_dataset=valid_set,
        data_collator=collate_fn if not args.rl else collate_fn_rl,
    )
    trainer.add_callback(EvaluateFirstStepCallback())

    trainer.train(resume_from_checkpoint=args.resume)
