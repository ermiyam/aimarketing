import unittest
import torch
from advanced_ai_system import AdvancedNeuralNetwork

class TestAdvancedAI(unittest.TestCase):
    def setUp(self):
        # Initialize the system with test parameters
        self.input_size = 10
        self.hidden_sizes = [64, 32]
        self.output_size = 2
        self.system = AdvancedNeuralNetwork(self.input_size, self.hidden_sizes, self.output_size)

    def test_initialization(self):
        print("\nTesting system initialization...")
        self.assertEqual(self.system.input_size, self.input_size)
        self.assertEqual(self.system.hidden_sizes, self.hidden_sizes)
        self.assertEqual(self.system.output_size, self.output_size)
        print("System initialized successfully")

    def test_forward_pass(self):
        print("\nTesting forward pass...")
        # Create a test input tensor
        batch_size = 5
        x = torch.randn(batch_size, self.input_size)
        output = self.system(x)
        
        # Check output shape
        self.assertEqual(output.shape, (batch_size, self.output_size))
        print(f"Forward pass successful. Output shape: {output.shape}")

    def test_training_history(self):
        print("\nTesting training history...")
        # Check if history dictionary is initialized correctly
        self.assertIn('loss', self.system.history)
        self.assertIn('accuracy', self.system.history)
        self.assertIn('adaptability', self.system.history)
        print("Training history initialized correctly")

    def test_weight_initialization(self):
        print("\nTesting weight initialization...")
        # Check if weights are initialized for all linear layers
        for layer in self.system.layers:
            if isinstance(layer, torch.nn.Linear):
                # Check if weights are not zero and properly initialized
                self.assertFalse(torch.all(layer.weight == 0))
                self.assertTrue(torch.isfinite(layer.weight).all())
                if layer.bias is not None:
                    self.assertTrue(torch.all(layer.bias == 0))
        print("Weights initialized correctly")

    def test_layer_structure(self):
        print("\nTesting layer structure...")
        # Count the number of layers
        linear_layers = sum(1 for layer in self.system.layers if isinstance(layer, torch.nn.Linear))
        batch_norm_layers = sum(1 for layer in self.system.layers if isinstance(layer, torch.nn.BatchNorm1d))
        relu_layers = sum(1 for layer in self.system.layers if isinstance(layer, torch.nn.ReLU))
        dropout_layers = sum(1 for layer in self.system.layers if isinstance(layer, torch.nn.Dropout))

        # We should have len(hidden_sizes) + 1 linear layers (including output)
        self.assertEqual(linear_layers, len(self.hidden_sizes) + 1)
        # We should have len(hidden_sizes) batch norm layers (one for each hidden layer)
        self.assertEqual(batch_norm_layers, len(self.hidden_sizes))
        # We should have len(hidden_sizes) ReLU layers
        self.assertEqual(relu_layers, len(self.hidden_sizes))
        # We should have len(hidden_sizes) dropout layers
        self.assertEqual(dropout_layers, len(self.hidden_sizes))

        print(f"Layer structure verified: {linear_layers} linear, {batch_norm_layers} batch norm, {relu_layers} ReLU, {dropout_layers} dropout layers")

if __name__ == '__main__':
    unittest.main(verbosity=2) 