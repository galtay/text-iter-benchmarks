import json
import os
import shutil
from datasets import load_dataset, load_from_disk
from datasets import Dataset
from glob import glob
import pandas as pd
from streaming import MDSWriter, StreamingDataset
from time import time
from tqdm import tqdm

class MosaicDataset(StreamingDataset):
    def __init__(self, local, remote):
        super().__init__(local, remote)

    def __getitem__(self, idx: int):
        obj = super().__getitem__(idx)
        return obj

dataset_name = "the_pile"

if dataset_name == "the_pile":
    FIELDS = {
        "text": "str",
    }
    TEXT_COL = "text"
    MAX_SAMPLES = 60_000
    ds = load_dataset("the_pile", "pubmed_central", split="train", streaming=True)

elif dataset_name == "pmcoa":
    FIELDS = {
        "text": "str",
        "pmid": "str",
        "accession_id": "str",
        "license": "str",
        "last_updated": "str",
        "retracted": "str",
        "citation": "str",
        "decoded_as": "str",
        "journal": "str",
        "doi": "str",
        "oa_subset": "str",
    }
    TEXT_COL = "text"
    MAX_SAMPLES = 60_000
    ds = load_dataset("gabrielaltay/pmcoa", split="validation", streaming=True)

else:
    raise ValueError()





if os.path.exists("samples.json"):
    with open("samples.json", "r") as fp:
        samples = json.load(fp)
else:
    samples = []
    for ii, sample in enumerate(ds):
        samples.append(sample)
        if ii + 1 >= MAX_SAMPLES:
            break
    with open("samples.json", "w") as fp:
        json.dump(samples, fp)


def write_hf_dataset(samples):
    if os.path.exists("hf"):
        shutil.rmtree("hf")
    df = pd.DataFrame(samples)
    hf_ds = Dataset.from_pandas(df)
    hf_ds.save_to_disk("hf")

def write_parquet_dataset(samples, samp_per_chunk=5_000):
    if os.path.exists("pq"):
        shutil.rmtree("pq")
    os.makedirs("pq", exist_ok=True)
    df = pd.DataFrame(samples)
    for ii in range(0, MAX_SAMPLES, samp_per_chunk):
        df1 = df.iloc[ii: ii+samp_per_chunk]
        df1.to_parquet(f"pq/chunk_{ii}.parquet")

def write_mosaic_dataset(samples, fields=FIELDS):
    if os.path.exists("mds"):
        shutil.rmtree("mds")
    mds_size_limit = 2**20 * 64
    with MDSWriter("mds", fields, size_limit=mds_size_limit) as out:
        for sample in samples:
            sample = {k:sample[k] for k in fields.keys()}
            for k in fields.keys():
                if sample[k] is None:
                    sample[k] = ""
                if isinstance(sample[k], str):
                    sample[k] = sample[k].encode("utf-8")
            out.write(sample)



write_hf_dataset(samples)
write_parquet_dataset(samples)
write_mosaic_dataset(samples)



def time_iter_mds():
    mds = MosaicDataset(local="mds", remote=None)
    num_samples = 0
    for sample in mds:
        num_samples += 1


def time_iter_hf(batch_size=10):
    hf_ds = load_from_disk("hf")
    # column removal has a large effect on timing
    hf_ds = hf_ds.remove_columns([col for col in hf_ds.column_names if col != TEXT_COL])
    num_samples = 0
    for samples in hf_ds.iter(batch_size=batch_size):
        num_samples += len(samples[TEXT_COL])

class PqDataset:

    COLUMNS = list(FIELDS.keys())

    def __init__(self, paths):
        self.paths = paths

    def iter_batch(self, batch_size=10, keep_cols=COLUMNS):
        for path in self.paths:
            df = pd.read_parquet(path, columns=keep_cols)
            for ii in range(0, df.shape[0], batch_size):
                df_batch = df.iloc[ii:ii+batch_size]
                yield df_batch

    def iter_text(self, text_col=TEXT_COL, batch_size=10):
        for df_batch in self.iter_batch(batch_size=batch_size, keep_cols=[text_col]):
            yield list(df_batch[text_col].values)


def time_iter_pq(batch_size=10):
    paths = glob("pq/*.parquet")
    pq_ds = PqDataset(paths)
    iter_text = pq_ds.iter_text(batch_size=batch_size)
    num_samples = 0
    for batch in iter_text:
        num_samples += len(batch)

n_trials=5

timings = {"mds": [], "hf": [], "pq": []}
for i in range(n_trials):

    t0 = time()
    time_iter_mds()
    timings["mds"].append(time()-t0)

    t0 = time()
    time_iter_hf(batch_size=10)
    timings["hf"].append(time()-t0)

    t0 = time()
    time_iter_pq(batch_size=10)
    timings["pq"].append(time()-t0)

for k,v in timings.items():
    print(k, v)
