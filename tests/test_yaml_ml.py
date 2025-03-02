import unittest
import glob
from utils import run_main


class TestYamlML(unittest.TestCase):
    tests_configs_folder = "./tests_configs"

    loading_test_configs_folder = f"{tests_configs_folder}/loading"
    preprocessing_test_configs_folder = f"{tests_configs_folder}/preprocessing"
    model_test_configs_folder = f"{tests_configs_folder}/model"
    parallel_test_configs_folder = f"{tests_configs_folder}/parallel"

    def test_loading(self):
        print("🚧 Testing dataset loading functionalities...")
        for cfg in glob.glob(f"{self.loading_test_configs_folder}/*.yaml"):
            run_main("--cfg", cfg)
            print(f"=> ✅ Run successfully for {cfg}")
        print("... ✅ Passed `test_loading`!")

    def test_preprocessing(self):
        print("🚧 Testing dataset preprocessing functionalities...")
        for cfg in glob.glob(f"{self.preprocessing_test_configs_folder}/*.yaml"):
            run_main("--cfg", cfg)
            print(f"=> ✅ Run successfully for {cfg}")
        print("... ✅ Passed `test_preprocessing`!")

    def test_model(self):
        print("🚧 Testing ML models (this might take a while)...")
        subfolders = [f"{self.model_test_configs_folder}/{t}" for t in ["regression", "classification"]]
        for folder in subfolders:
            for cfg in glob.glob(f"{folder}/*.yaml"):
                run_main("--cfg", cfg)
                print(f"=> ✅ Run successfully for {cfg}")
        print("... ✅ Passed `test_model`!")

    def test_parallel(self):
        n_processes = 2
        print(f"🚧 Testing parallel computing (multiprocessing) with {n_processes} worker processes...")
        run_main("--cfg", self.parallel_test_configs_folder, "--n_processes", "2")
        print("... ✅ Passed `test_parallel`!")


if __name__ == "__main__":
    unittest.main()