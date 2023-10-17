from transformers import ViTImageProcessor, AutoTokenizer, VisionEncoderDecoderModel, AutoProcessor, AutoModel, CLIPTokenizer
from transformers.models.vision_encoder_decoder.modeling_vision_encoder_decoder import shift_tokens_right, CrossEntropyLoss
from torch import nn
from transformers.modeling_outputs import BaseModelOutputWithPooling, Seq2SeqLMOutput
import torch


def compute_loss(model, pixel_values, labels, sample_weights):
    output = model(pixel_values=pixel_values, labels=labels)

    criterion = nn.CrossEntropyLoss(reduction='none')

    loss = criterion(
        output.logits[:, :-1, :].reshape(-1, 50257),
        labels[:, 1:].reshape(-1))

    loss = loss.reshape(output.logits.size(dim=0), -1)
    sample_weights = sample_weights.squeeze(0).unsqueeze(1).repeat(1, output.logits.size(dim=1) - 1)

    loss = loss * sample_weights
    return loss.mean()


def compute_rl_loss(model, image_processor, tokenizer, pixel_values, gt_labels, reward_fct):

    generated_ids = model.generate(
        pixel_values,
        max_new_tokens=40,
        num_beams=5,
        no_repeat_ngram_size=2,
        early_stopping=True,
        num_return_sequences=3,
    )
    generated_texts = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)
    score = reward_fct(generated_texts, pixel_values, model.device)  # return tensor
    mu = score.mean()
    sample_weights = score - mu

    return compute_loss(model, pixel_values, generated_ids, sample_weights)


def compute_image_representation_from_image_instance(pixel_values, clip_model):
    visual_outputs = clip_model.vision_model(pixel_values=pixel_values)
    image_embeds = visual_outputs[1]
    image_embeds = clip_model.visual_projection(image_embeds)  # [1 x embed_dim]
    image_embeds = image_embeds / image_embeds.norm(dim=-1, keepdim=True)
    return image_embeds


def compute_image_text_similarity_via_embeddings(image_embeds, text_embeds, clip_model):
    '''
        image_embeds: batch x embed_dim
        text_embeds: batch x len(text_list) x embed_dim
    '''
    text_embeds = text_embeds.view(image_embeds.shape[0], -1, text_embeds.shape[-1])
    text_embeds = text_embeds / text_embeds.norm(dim=-1, keepdim=True)
    image_embeds = image_embeds.unsqueeze(-1)
    logit_scale = clip_model.logit_scale.exp()
    logits_per_text = torch.matmul(text_embeds, image_embeds) * logit_scale
    logits_per_image = logits_per_text.squeeze(-1)
    return logits_per_image.softmax(dim=1)  # , logits_per_image/logit_scale # batch x len(text_list)


def compute_text_representation(text_list, clip_model, clip_tokenizer, device):
    # text_list: a list of text
    text_inputs = clip_tokenizer(text_list, padding=True, return_tensors="pt",
                                 max_length=clip_tokenizer.max_len_single_sentence + 2, truncation=True).to(device)
    # self.tokenizer.max_len_single_sentence + 2 = 77
    input_ids, attention_mask = text_inputs['input_ids'], text_inputs['attention_mask']

    text_outputs = clip_model.text_model(
        input_ids=input_ids,
        attention_mask=attention_mask
    )
    text_embeds = text_outputs[1]
    text_embeds = clip_model.text_projection(text_embeds)
    return text_embeds


def compute_image_text_similarity_via_raw_text(image_embeds, text_list, clip_model, clip_tokenizer, device):
    text_embeds = compute_text_representation(text_list, clip_model, clip_tokenizer, device)
    return compute_image_text_similarity_via_embeddings(image_embeds, text_embeds, clip_model)


def reward_clip(clip_model, clip_tokenizer):
    def _reward_fct(generated_texts, pixel_values, device):
        image_embeds = compute_image_representation_from_image_instance(pixel_values, clip_model)

        return compute_image_text_similarity_via_raw_text(image_embeds, generated_texts, clip_model, clip_tokenizer, device)

    return _reward_fct


class RLVisionEncoderDecoderModel(VisionEncoderDecoderModel):
    def __init__(self, rl, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.rl = rl
        if self.rl:
            self.clip_model = AutoModel.from_pretrained("openai/clip-vit-base-patch32")
            self.clip_tokenizer = AutoTokenizer.from_pretrained("openai/clip-vit-base-patch32")
            self.tokenizer = AutoTokenizer.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.image_processor = ViTImageProcessor.from_pretrained("nlpconnect/vit-gpt2-image-captioning")
            self.loss_fct = reward_clip(self.clip_model, self.clip_tokenizer)
        else:
            self.loss_fct = CrossEntropyLoss()

    def forward(
            self,
            pixel_values=None,
            decoder_input_ids=None,
            decoder_attention_mask=None,
            encoder_outputs=None,
            past_key_values=None,
            decoder_inputs_embeds=None,
            labels=None,
            use_cache=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
            **kwargs,
    ):
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict
        kwargs_encoder = {argument: value for argument, value in kwargs.items() if not argument.startswith("decoder_")}

        kwargs_decoder = {
            argument[len("decoder_"):]: value for argument, value in kwargs.items() if argument.startswith("decoder_")
        }
        if encoder_outputs is None:
            encoder_outputs = self.encoder(
                pixel_values,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
                **kwargs_encoder,
            )
        encoder_hidden_states = encoder_outputs[0]
        # torch.save(encoder_hidden_states, 'encoder_hidden_states.pth')

        if (
                self.encoder.config.hidden_size != self.decoder.config.hidden_size
                and self.decoder.config.cross_attention_hidden_size is None
        ):
            encoder_hidden_states = self.enc_to_dec_proj(encoder_hidden_states)

        # else:
        encoder_attention_mask = None

        if (labels is not None) and (decoder_input_ids is None and decoder_inputs_embeds is None):
            decoder_input_ids = shift_tokens_right(
                labels, self.config.pad_token_id, self.config.decoder_start_token_id
            )

        # Decode
        decoder_outputs = self.decoder(
            input_ids=decoder_input_ids,
            attention_mask=decoder_attention_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            inputs_embeds=decoder_inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            use_cache=use_cache,
            past_key_values=past_key_values,
            return_dict=return_dict,
            **kwargs_decoder,
        )

        # Compute loss independent from decoder (as some shift the logits inside them)
        loss = None
        if labels is not None:
            if self.rl:
                loss = compute_rl_loss(
                    self, self.image_processor, self.tokenizer,
                    pixel_values, labels,
                    self.loss_fct
                )
            else:
                logits = decoder_outputs.logits if return_dict else decoder_outputs[0]
                loss = self.loss_fct(logits.reshape(-1, self.decoder.config.vocab_size), labels.reshape(-1))

        if not return_dict:
            if loss is not None:
                return (loss,) + decoder_outputs + encoder_outputs
            else:
                return decoder_outputs + encoder_outputs

        return Seq2SeqLMOutput(
            loss=loss,
            logits=decoder_outputs.logits,
            past_key_values=decoder_outputs.past_key_values,
            decoder_hidden_states=decoder_outputs.hidden_states,
            decoder_attentions=decoder_outputs.attentions,
            cross_attentions=decoder_outputs.cross_attentions,
            encoder_last_hidden_state=encoder_outputs.last_hidden_state,
            encoder_hidden_states=encoder_outputs.hidden_states,
            encoder_attentions=encoder_outputs.attentions,
        )
