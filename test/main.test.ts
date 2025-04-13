import { describe, it, expect } from 'vitest'
import add from '../src/main';
import Neuron from '../src/Neuron';
import Layer from '../src/Layer';

describe('add()', () => {
    it('should add two number', () => {
        expect(add(2, 3)).toBe(5)
    })
});

describe('weightedSum()', () =>{
    it('should do the weight sum and activation function of the neuron', () => {
        expect(new Neuron(1, [1, 1, 1]).weightedSum([1,2,3], [true, true, true])).toBeCloseTo(0.99988, 5)
    })
});

describe('layerConstructor()', () =>{
    it('should do create a layer where each element is a tuple with [Neuron, number[]]', () => {
        const layer = new Layer([new Neuron(1,[1])]);

        for (const [neuron, numbers] of layer.neurons) {
            expect(neuron).toBeInstanceOf(Neuron);
            expect(Array.isArray(numbers)).toBe(true);
            expect(numbers.every(n => typeof n === "number")).toBe(true);
        }
    })
});