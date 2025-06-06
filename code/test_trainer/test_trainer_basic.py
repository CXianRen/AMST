from common import BasicTrainer
from baseline.ogm3_new import OGMTrainer
from baseline.mslr3_new import MSLRTrainer
from baseline.pmr3_new import PMRTrainer
from baseline.mla_new import MLATrainer
from mart import MARTTrainer

from metrics import set_profiler

from .test_common import Mock_dataset, run_trainer_test

import os,sys

class TrainerHacker:
    def init_logging(self):
        self.save_path = "/tmp/"
        self.tsb_writer = None
        self.prof = set_profiler("", False)

    def init_dataset(self):
        size = getattr(self, 'mock_data_size', 64)
        self.train_dataset = Mock_dataset(size)
        self.val_dataset = Mock_dataset(size)
        self.test_dataset = Mock_dataset(size)
        self.n_classes = 6

class TestBasicTrainer(TrainerHacker, BasicTrainer):
    def __init__(self):
        super().__init__("--epochs 1 --batch_size 2".split())

class TestOGMTrainer(TrainerHacker, OGMTrainer):
    def __init__(self):
        super().__init__("--epochs 1 --batch_size 2".split())
                
class TestMSLRTrainer(TrainerHacker, MSLRTrainer):
    def __init__(self):
        super().__init__("--epochs 1 --batch_size 2".split())
        
class TestPMRTrainer(TrainerHacker, PMRTrainer):
    def __init__(self):
        super().__init__("--epochs 1 --batch_size 2".split())

class TestMLATrainer(TrainerHacker, MLATrainer):
    def __init__(self):
        super().__init__("--epochs 1 --batch_size 2".split())

class TestMARTTrainer(TrainerHacker, MARTTrainer):
    def __init__(self):
        super().__init__("--epochs 1 --batch_size 2".split())


if __name__ == "__main__":
    run_trainer_test(TestBasicTrainer, "test_basic_trainer")
    run_trainer_test(TestOGMTrainer, "test_ogm_trainer")
    run_trainer_test(TestMSLRTrainer, "test_mslr3_trainer")
    run_trainer_test(TestPMRTrainer, "test_pmr3_trainer")
    run_trainer_test(TestMLATrainer, "test_mla_trainer")
    run_trainer_test(TestMARTTrainer, "test_mart_trainer")