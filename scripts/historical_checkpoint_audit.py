"""Audit original v6/v7 confirmation predictions against original training labels.

Historical validation is adaptive: neither version is an untouched Kaggle test.
Run:
 python -m scripts.historical_checkpoint_audit --data data/structural-damage-classification.zip --checkpoint PATH_TO_ARCHIVE.zip
"""
import argparse
import hashlib
import io
import json
from pathlib import Path
from zipfile import ZipFile
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score

EXPECTED_SHA="231c9bf0430faa9c1d2281b4edbcd9ce0d7a863c6ff67042a8fbc992469d37bb"
CLASSES=[1,2,3,4,5]

def audit_version(checkpoint, truth, version, output):
    """Validate indices, labels and all fold rows, then export matrices and plots."""
    prefix="research_"+version
    with ZipFile(checkpoint) as z:
        with np.load(io.BytesIO(z.read(prefix+"/confirmation/pooled.npz"))) as a:
            ids=a["indices"]; y=a["truth"]+1; pred=a["prediction"]+1
        assert len(ids)==60000 and len(np.unique(ids))==60000
        assert np.array_equal(truth[ids],y)
        folds=[]
        for k in range(3):
            with np.load(io.BytesIO(z.read(prefix+f"/confirmation/fold_{k}.npz"))) as f:
                idx, yy, pp=f["indices"],f["truth"]+1,f["prediction"]+1
                assert np.array_equal(truth[idx],yy)
                folds.append((idx,yy,pp))
        for j,pool in enumerate((ids,y,pred)):
            assert np.array_equal(np.concatenate([v[j] for v in folds]),pool)
    cm=confusion_matrix(y,pred,labels=CLASSES)
    report={"source":prefix+"/confirmation/pooled.npz","n":len(y),
            "macro_f1":float(f1_score(y,pred,labels=CLASSES,average="macro")),
            "accuracy":float(accuracy_score(y,pred)),"matrix":cm.tolist(),
            "per_class":classification_report(y,pred,labels=CLASSES,output_dict=True,zero_division=0),
            "limitation":"Adaptive historical confirmation; not untouched holdout."}
    (output/(version+"_metrics.json")).write_text(json.dumps(report,indent=2))
    pd.DataFrame(cm,index=CLASSES,columns=CLASSES).to_csv(output/(version+"_counts.csv"))
    for norm in (False,True):
        values=cm / cm.sum(1,keepdims=True) if norm else cm
        fig,ax=plt.subplots(figsize=(8,7))
        ax.imshow(values,cmap="Blues")
        ax.set_xticks(range(5),CLASSES);ax.set_yticks(range(5),CLASSES)
        ax.set(xlabel="Predicted grade",ylabel="Actual grade",title=f"Historical {version}")
        for i in range(5):
            for j in range(5):
                ax.text(j,i,(f"{values[i,j]:.1%}" if norm else str(cm[i,j])),ha="center",va="center")
        fig.tight_layout()
        fig.savefig(output/(version+("_normalized.png" if norm else "_counts.png")),dpi=160)
        plt.close(fig)
    return report

def main():
    """Hash source labels before computing independent per-row arithmetic."""
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data",required=True)
    parser.add_argument("--checkpoint",required=True)
    parser.add_argument("--output",default="artifacts/historical_audit")
    args=parser.parse_args()
    output=Path(args.output);output.mkdir(parents=True,exist_ok=True)
    with ZipFile(args.data) as z: raw=z.read("train.csv")
    if hashlib.sha256(raw).hexdigest()!=EXPECTED_SHA: raise ValueError("Train file checksum mismatch")
    frame=pd.read_csv(io.BytesIO(raw),usecols=["damage_grade"])
    truth=frame.damage_grade.str.extract(r"([1-5])")[0].astype(int).to_numpy()
    result={v:audit_version(args.checkpoint,truth,v,output) for v in ("v6","v7")}
    (output/"summary.json").write_text(json.dumps(result,indent=2))
    print({v:(r["n"],r["macro_f1"]) for v,r in result.items()})

if __name__=="__main__": main()
