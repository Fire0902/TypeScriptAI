import { describe, it, expect } from 'vitest';
import Neuron from '../src/Neuron';

describe('Neuron', () => {
    it('should create a neuron with the correct properties', () => {
        const neuron = new Neuron(1, [0.5, 0.2, -0.3]);
        expect(neuron.bias).toBe(1);
        expect(neuron.weight).toEqual([0.5, 0.2, -0.3]);
        expect(neuron.result).toBe(-1);
        expect(neuron.entrys).toEqual([]);
        expect(neuron.learningRate).toBe(1);
        expect(neuron.error).toBe(-1);
    });

    it('should calculate the activation function correctly', () => {
        const neuron = new Neuron(0, [0]);
        const result = neuron.activationFunction(0); // sigmoid(0) should return 0.5
        expect(result).toBe(0.5);
    });

    it('should calculate the sigmoid derivative correctly', () => {
        const neuron = new Neuron(0, [0]);
        neuron.result = 0.5; // Let's assume the result is 0.5
        const derivative = neuron.sigmoidDerivative();
        expect(derivative).toBeCloseTo(0.25, 5); // 0.5 * (1 - 0.5) = 0.25
    });

    it('should calculate weighted sum correctly', () => {
        const neuron = new Neuron(0, [0.5, 0.2, -0.3]);
        const result = neuron.weightedSum([1, 2, 3], [true, true, true]);
        expect(result).toBeCloseTo(0.4, 5); // Weighted sum of the inputs with weights
    });
});
