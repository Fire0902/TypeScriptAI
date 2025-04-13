import { describe, it, expect } from 'vitest';
import NeuralNetwork from '../src/NeuralNetwork';
import Layer from '../src/Layer';
import Neuron from '../src/Neuron';

describe('NeuralNetwork', () => {
    it('should create a neural network with layers', () => {
        const layer1 = new Layer([new Neuron(1, [0.1, 0.2]), new Neuron(1, [0.3, 0.4])]);
        const layer2 = new Layer([new Neuron(1, [0.5, 0.6])]);
        const nn = new NeuralNetwork([layer1, layer2]);
        expect(nn.layers).toHaveLength(2); // Should have 2 layers
    });

    it('should compute the error value using MSE', () => {
        const layer1 = new Layer([new Neuron(1, [0.1, 0.2])]);
        const layer2 = new Layer([new Neuron(0, [0.3, 0.4])]);
        const nn = new NeuralNetwork([layer1, layer2]);
        const trainingData: Array<[number[][], number]> = [
            [
                [[1, 2], [3, 4]],
                5
            ],
            [
                [[1, 0], [2, 1]],
                3
            ]
        ];
        
        nn.errorValue((results: Array<[number, number]>) => {
            const loss = nn.meanSquaredError(results);
            expect(loss).toBeCloseTo(25, 2); // MSE = ((prediction - target)^2)/n = ((0 - 5)^2)/1 = 25
        })(trainingData);
    });

    it('should update weights during backpropagation', () => {
        const layer1 = new Layer([new Neuron(1, [0.1, 0.2])]);
        const layer2 = new Layer([new Neuron(0, [0.3, 0.4])]);
        const nn = new NeuralNetwork([layer1, layer2]);
        const trainingData: Array<[number[][], number]> = [
            [
                [[1, 2], [3, 4]],
                5
            ],
            [
                [[1, 0], [2, 1]],
                3
            ]
        ];
        
        nn.errorValue((results: Array<[number, number]>) => {
            const loss = nn.meanSquaredError(results);
            expect(loss).toBeCloseTo(25, 2); // MSE calculation as before
        })(trainingData);

        // Before backpropagation, we check that the weight is correctly initialized
        const initialWeight = layer1.neurons[0][0].weight[0];

        // Perform backpropagation (simplified for this example)
        nn.backTracking();

        // Ensure that weights have been updated (in a real network, weights would be updated by a small amount)
        expect(layer1.neurons[0][0].weight[0]).not.toBe(initialWeight); // The weight should change after backpropagation
    });

    it('should propagate errors correctly during backpropagation', () => {
        const layer1 = new Layer([new Neuron(1, [0.1, 0.2])]);
        const layer2 = new Layer([new Neuron(0, [0.3, 0.4])]);
        const nn = new NeuralNetwork([layer1, layer2]);
        const trainingData: Array<[number[][], number]> = [
            [
                [[1, 2], [3, 4]],
                5
            ],
            [
                [[1, 0], [2, 1]],
                3
            ]
        ];
        

        nn.errorValue((results: Array<[number, number]>) => {
            const loss = nn.meanSquaredError(results);
            expect(loss).toBeCloseTo(25, 2);
        })(trainingData);

        // Run backtracking
        nn.backTracking();

        // Check that the error has been propagated properly
        expect(layer1.neurons[0][0].error).not.toBe(-1); // The error should have been computed and propagated
    });
});
