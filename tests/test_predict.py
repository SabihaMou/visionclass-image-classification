import unittest
from PIL import Image
from visionclass.predict import prepare_image, top_predictions


class TestImageProcessing(unittest.TestCase):
    def test_rgb_conversion(self):
        self.assertEqual(prepare_image(Image.new("L", (8, 8))).mode, "RGB")

    def test_dimensions_preserved(self):
        self.assertEqual(prepare_image(Image.new("RGBA", (12, 9))).size, (12, 9))


class TestPredictions(unittest.TestCase):
    def test_top_prediction_sorted(self):
        import torch
        results = top_predictions(torch.tensor([[0.1, 2.0, 1.0]]), ["cat", "dog", "bird"], k=2)
        self.assertEqual([x["label"] for x in results], ["dog", "bird"])
        self.assertGreater(results[0]["confidence"], results[1]["confidence"])

    def test_k_capped_to_class_count(self):
        import torch
        self.assertEqual(len(top_predictions(torch.tensor([[0.0, 1.0]]), ["a", "b"], k=5)), 2)

    def test_invalid_k(self):
        import torch
        with self.assertRaises(ValueError):
            top_predictions(torch.tensor([[0.0]]), ["a"], k=0)


if __name__ == "__main__":
    unittest.main()
