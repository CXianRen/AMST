# The basic trainer is the naive method without any optimization.
from common import BasicTrainer

if __name__ == "__main__":
    trainer = BasicTrainer()
    trainer.train_validate()