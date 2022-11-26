_CITATION = """
None
"""

_DESCRIPTION = """
None
"""

_KWARGS_DESCRIPTION = """
None
"""

from sklearn.metrics import f1_score, accuracy_score
import datasets

class MyMetric(datasets.Metric):
    def _info(self):
        return datasets.MetricInfo(
            description=_DESCRIPTION,
            citation=_CITATION,
            inputs_description=_KWARGS_DESCRIPTION,
            features=datasets.Features(
                {
                    "predictions": datasets.Sequence(datasets.Value("int32")),
                    "references": datasets.Sequence(datasets.Value("int32")),
                }
                if self.config_name == "multilabel"
                else {
                    "predictions": datasets.Value("int32"),
                    "references": datasets.Value("int32"),
                }
            ),
        )

    def _compute(self, predictions, references):
        micro_f1 = f1_score(references, predictions, average='micro')
        macro_f1 = f1_score(references, predictions, average='macro')
        accuracy = accuracy_score(references, predictions)
        return {
            "micro_f1": float(micro_f1) if micro_f1.size == 1 else micro_f1,
            "macro_f1": float(macro_f1) if macro_f1.size == 1 else macro_f1,
            "accuracy": float(accuracy) if accuracy.size == 1 else accuracy,
        }