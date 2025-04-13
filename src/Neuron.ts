class Neuron {
    bias: number;
    weight: number[];
    result: number;
    entrys: number[];
    learningRate: number;
    error: number;

    constructor(bias: number, weight: number[]){
        this.bias = bias;
        this.weight = weight;
        this.result = -1;
        this.entrys = []
        this.learningRate = 1;
        this.error = -1;
    }
    
    // this function do the activation function of the neuron
    // 1/(1 + exp(-x)) (sigmoid)
    activationFunction(entry: number){
        return 1/(1 + Math.exp(-entry))
    }

    //σ`(x) = σ(x) * (1-σ(x))
    sigmoidDerivative(){
        const sigmoid: number = this.activationFunction(this.result);
        return sigmoid * (1-sigmoid);
    }

    weightedSum(entrys: number[], masks: boolean[]){
        this.entrys = entrys;
        let sum = 0;
        entrys.forEach((entry, index) => {
            if(masks[index] !== false)
                sum += (entry * this.weight[entrys.indexOf(entry)]) + this.bias
        });
        const result = this.activationFunction(sum);
        this.result = result;
        return result;
    }
    weightedSumWithoutActivation(masks: boolean[]){
        let sum = 0;
        this.entrys.forEach((entry, index) => {
            if(masks[index] !== false)
                sum += (entry * this.weight[this.entrys.indexOf(entry)]) + this.bias
        });
        return sum;
    }

    empty(){
        this.result = -1;
        this.entrys = []
        this.error = -1;
    }
    
}
export default Neuron;