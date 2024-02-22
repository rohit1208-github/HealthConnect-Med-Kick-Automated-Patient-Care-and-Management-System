import pandas as pd
from fastai.text.all import *
import blurr
from blurr.text.data.all import *
from blurr.text.modeling.all import *
import pickle
import nltk
import datasets

dataset = load_dataset('ccdv/pubmed-summarization')

train_df = pd.DataFrame(dataset['train'])
train_df['article'] = train_df['article'].str.replace('\n', '')
train_df = train_df[:1000]

model_name = "google/pegasus-large"
arch, config, tokenizer, model = get_hf_objects(model_name, model_cls=PegasusForConditionalGeneration)

preproc = SummarizationPreprocessor(tokenizer, id_attr="id", text_attr="article", target_text_attr="abstract",
                                    max_input_tok_length=1024, max_target_tok_length=256, min_summary_char_length=30)
proc_df = preproc.process_df(train_df)

text_kwargs = default_text_gen_kwargs(config, model, task="summarization")
batch_tok_transform = Seq2SeqBatchTokenizeTransform(arch, config, tokenizer, model, text_gen_kwargs=text_kwargs)

blocks = (Seq2SeqTextBlock(batch_tokenize_tfm=batch_tok_transform), noop)
dblock = DataBlock(blocks=blocks, get_x=ColReader("proc_article"), get_y=ColReader("proc_abstract"),
                   splitter=RandomSplitter())
dls = dblock.dataloaders(proc_df, bs=2)

metrics = {"rouge": {"compute_kwargs": {"rouge_types": ["rouge1", "rouge2", "rougeL"], "use_stemmer": True},
                     "returns": ["rouge1", "rouge2", "rougeL"]},
           "bertscore": {"compute_kwargs": {"lang": "en"}, "returns": ["precision", "recall", "f1"]},
           "bleu": {"returns": "bleu"}, "meteor": {"returns": "meteor"}, "sacrebleu": {"returns": "score"}}

wrapped_model = BaseModelWrapper(model)
learn = Learner(dls, wrapped_model, loss_func=PreCalculatedCrossEntropyLoss(), opt_func=partial(Adam), cbs=[BaseModelCallback],
                splitter=partial(blurr_seq2seq_splitter, arch=arch)).to_fp16()

lr_suggestion = learn.lr_find(suggest_funcs=[minimum, steep, valley, slide])

nltk.download('punkt')

gc.collect()
torch.cuda.empty_cache()

learn.fit_one_cycle(10, lr_max=lr_suggestion.valley, cbs=[Seq2SeqMetricsCallback(custom_metrics=metrics, calc_every="epoch")])

learn.show_results()
learn.metrics = None
learn = learn.to_fp32()
learn.export(fname="pegasus_summary_export.pkl")
