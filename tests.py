import Main
import unittest
import numpy as np


class TestForwardQ1(unittest.TestCase):

    def test_initialize_parameters(self):
        # test_in = (1, 1)
        # ans = Main.initialize_parameters(test_in)
        # expected_weights = [np.zeros((1, 1))] * 2
        # expected_bias_len = (2, 1)
        # for ans_w, expected_w in zip(ans['W'], expected_weights):
        #     self.assertTupleEqual(ans_w.shape, expected_w.shape)
        # self.assertEqual(expected_bias_len, len(ans['b']))

        test_in = (4, 2, 10)
        ans = Main.initialize_parameters(test_in)
        expected_weights = [(4, 2), (2, 10)]
        expected_bias = [(2, 1), (10, 1)]
        for ans_w, expected_w in zip(ans['W'], expected_weights):
            self.assertTupleEqual(ans_w.shape, expected_w)
        for ans_b, expected_b in zip(ans['b'], expected_bias):
            self.assertTupleEqual(ans_b.shape, expected_b)

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

    def test_batchnorm(self):
        a = np.array([[3, 5, 1], [1, 2, 3]])
        ans = Main.apply_batchnorm(a)
        expected_ans = [[0., 1.2245152962941819, -1.2245152962941819],
                        [-1.2238273448265007, 0., 1.2238273448265007]]
        self.assertListEqual(ans.tolist(), expected_ans)

    def test_L_model_forward(self):
        inp = np.random.randn(2, 3)
        inp_flat = np.reshape(inp, (6, 1))
        init = Main.initialize_parameters((6, 3, 10))
        ans_AL, ans_cache = Main.L_model_forward(inp_flat, init, False)
        self.assertTupleEqual(ans_AL.shape, (10, 1))
        self.assertEqual(len(ans_cache), 2)


if __name__ == '__main__':
    unittest.main()
