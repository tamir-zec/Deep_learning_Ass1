import Main
import unittest
import numpy as np


class TestForwardQ1(unittest.TestCase):

    def test_initialize_parameters(self):
        test_in = (1, 1)
        ans = Main.initialize_parameters(test_in)
        expected_weights = [np.zeros((1, 1))] * 2
        expected_bias_len = len(test_in)
        for ans_w, expected_w in zip(ans['W'], expected_weights):
            self.assertTupleEqual(ans_w.shape, expected_w.shape)
        self.assertEqual(expected_bias_len, len(ans['b']))

        test_in = (4, 2, 5, 8, 10)
        ans = Main.initialize_parameters(test_in)
        expected_weights = [np.zeros((1, x)) for x in test_in]
        expected_bias_len = len(test_in)
        for ans_w, expected_w in zip(ans['W'], expected_weights):
            self.assertTupleEqual(ans_w.shape, expected_w.shape)
        self.assertEqual(expected_bias_len, len(ans['b']))

    def test_linear_forward(self):
        W = np.array([0.5, 0.75])
        A = np.array([[1, 2], [3, 2]])
        ans_z, ans_cache = Main.linear_forward(A, W, 0.6)
        self.assertDictEqual(ans_cache, {
            'A': A,
            'W': W,
            'b': 0.6
        })
        self.assertListEqual([3.35, 3.1], ans_z.tolist())

    def test_softmax(self):
        z = np.array([[0.2, 0.1], [2, 3]])
        expected_ans = ([[0.14185106490048777, 0.05215356307841774],
                         [0.8581489350995122, 0.9478464369215823]])
        ans_a, ans_z = Main.softmax(z)
        self.assertListEqual(ans_a.tolist(), expected_ans)
        self.assertDictEqual(ans_z, {'Z': z})

    def test_relu(self):
        z = np.array([[0.2, -0.1], [2, 3], [-2, 3]])
        expected_ans = ([[0.2, 0.], [2, 3], [0., 3]])
        ans_a, ans_z = Main.relu(z)
        self.assertListEqual(ans_a.tolist(), expected_ans)
        self.assertDictEqual(ans_z, {'Z': z})

    def test_linear_activation_forward(self):
        A = np.array([[0.2, -0.1], [2, 3]])
        W = np.array([5, 6])
        expected_z = np.array([13.5, 16.75])
        bias = [0.5, -0.75]
        ans_a, ans_cache = Main.linear_activation_forward(A, W, bias, Main.softmax)
        self.assertListEqual(ans_a.tolist(), [0.03732688734412946, 0.9626731126558706])
        # self.assertDictEqual(ans_cache, {'A': A,
        #                                  'W': W,
        #                                  'b': bias,
        #                                  'Z': expected_z})


if __name__ == '__main__':
    unittest.main()
