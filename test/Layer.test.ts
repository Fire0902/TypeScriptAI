import { describe, it, expect } from 'vitest';
import Layer from '../src/Layer';
import Neuron from '../src/Neuron';

describe('Layer', () => {
    it('should create a layer with neurons and empty entry arrays', () => {
        const layer = new Layer([new Neuron(1, [0.1, 0.2, 0.3])]);
        expect(layer.neurons).toHaveLength(1);
        expect(layer.neurons[0][0]).toBeInstanceOf(Neuron); // Checking if it's a Neuron
        expect(layer.neurons[0][1]).toEqual([]); // Checking if entrys are empty
    });

    it('should set entries correctly', () => {
        const layer = new Layer([new Neuron(1, [0.5, -0.5])]);
        layer.setEntrys([[1, 2]]);
        expect(layer.neurons[0][1]).toEqual([1, 2]);
    });

    it('should activate neurons correctly', () => {
        const neuron1 = new Neuron(1, [0.5, -0.5]);
        const neuron2 = new Neuron(0, [0.3, 0.7]);
        const layer = new Layer([neuron1, neuron2]);
        layer.setEntrys([[1, 2], [3, 4]]);
        const result = layer.neuronsActivation();
        expect(result).toBeInstanceOf(Array); // Should return an array of results
        expect(result.length).toBe(2); // Should have as many results as neurons
    });
});
