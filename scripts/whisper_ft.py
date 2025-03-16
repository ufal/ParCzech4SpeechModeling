import hydra
from parczech.asr_utils.utils import (
    DataCollatorSpeechSeq2SeqWithPadding,
    MetricComputer,
    get_dataset,
)
from transformers import (
    Seq2SeqTrainer,
    WhisperForConditionalGeneration,
    WhisperProcessor,
)


class MySeq2SeqTrainer(Seq2SeqTrainer):
    def __init__(
        self,
        model=None,
        args=None,
        data_collator=None,
        train_dataset=None,
        eval_dataset=None,
        tokenizer=None,
        model_init=None,
        compute_metrics=None,
        callbacks=None,
        optimizers=(None, None),
        preprocess_logits_for_metrics=None,
        my_tokenizer=None,
    ):
        self.my_tokenizer = my_tokenizer
        super().__init__(
            model=model,
            args=args,
            data_collator=data_collator,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            tokenizer=tokenizer,
            model_init=model_init,
            compute_metrics=compute_metrics,
            callbacks=callbacks,
            optimizers=optimizers,
            preprocess_logits_for_metrics=preprocess_logits_for_metrics,
        )

    def prediction_step(self, model, inputs, prediction_loss_only, ignore_keys=None, **gen_kwargs):
        try:
            return super().prediction_step(model, inputs, prediction_loss_only, ignore_keys, **gen_kwargs)
        except Exception as e:
            # print(f"Error: {e}")
            print(inputs["input_features"].shape, inputs["labels"].shape, self.my_tokenizer.decode(inputs["labels"][0], skip_special_tokens=False))
            raise e

@hydra.main(config_path="../configs", config_name="whisper_ft")
def main(cfg):
    processor = WhisperProcessor.from_pretrained(
        cfg.model_type,
        language="Czech",
        task="transcribe"
    )

    dataset = get_dataset(
        cfg.dataset_params.name,
        cfg.dataset_params.sr,
        processor,
        cfg.dataset_params.num_proc
    )

    model = WhisperForConditionalGeneration.from_pretrained(cfg.model_type)
    model.generation_config.language = "czech"
    model.generation_config.task = "transcribe"

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(
        processor=processor,
        decoder_start_token_id=model.config.decoder_start_token_id,
    )

    metric_computer = MetricComputer(processor)

    training_args = hydra.utils.instantiate(cfg.training_args)
    trainer = MySeq2SeqTrainer(
        args=training_args,
        model=model,
        train_dataset=dataset["train"],
        eval_dataset=dataset["test"],
        data_collator=data_collator,
        compute_metrics=metric_computer,
        tokenizer=processor.feature_extractor,
        my_tokenizer=processor.tokenizer,
    )

    trainer.train()



if __name__ == "__main__":
    main()
