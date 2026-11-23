from django.conf import settings
from django.core.management.base import BaseCommand

from finmind.pipeline import train_all


class Command(BaseCommand):
    help = ("Train everything: BPE tokenizer, mini-GPT pretraining, LoRA SFT + DPO, quantization benchmark, "
            "speculative-decoding draft. Writes artifacts/ (~3 min on CPU).")

    def add_arguments(self, parser):
        parser.add_argument("--steps", type=int, default=250, help="pretraining steps")

    def handle(self, *args, **opts):
        art = settings.FINMIND_ARTIFACTS
        train_all(art, pretrain_steps=opts["steps"], log=lambda *a: self.stdout.write(" ".join(str(x) for x in a)))
        self.stdout.write(self.style.SUCCESS(f"Done. Metrics: {art / 'metrics.json'}"))
        self.stdout.write("Restart the server so it loads the new model.")
