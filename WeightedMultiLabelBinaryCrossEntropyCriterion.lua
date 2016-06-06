local WeightedMultiLabelBinaryCrossEntropyCriterion, parent = torch.class('nn.WeightedMultiLabelBinaryCrossEntropyCriterion', 'nn.Criterion')

local eps = 1e-12

function WeightedMultiLabelBinaryCrossEntropyCriterion:__init(counts, N, sizeAverage)
    parent.__init(self)
    if sizeAverage ~= nil then
        self.sizeAverage = sizeAverage
    else
        self.sizeAverage = true
    end
    assert(counts:dim() == 1, "counts input should be 1-D Tensor")
    assert(counts:size(1) > 0, "need counts to compute loss")
    self.counts = counts
    self.K = #counts
    self.N = N
    self.NNK = torch.mul(counts, -1):add(self.N) -- (N - Nk) compute once
end

function WeightedMultiLabelBinaryCrossEntropyCriterion:updateOutput(input, target)
    -- - log(input) * (target/counts[class]) + log(1 - input) * ((1 - target) / (N - counts[class]) 

    assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local output

    counts = self.counts
    if counts ~= nil and target:dim() ~= 1 then
        counts = counts:view(1, target:size(2)):expandAs(target)
    end
    NNK = self.NNK
    if NNK ~= nil and target:dim() ~= 1 then
        NNK = NNK:view(1, target:size(2)):expandAs(target)
    end

    buffer:resizeAs(input)

    -- log(input) * (target/counts[class])
    buffer:add(input, eps):log()
    output = torch.dot(torch.cdiv(target, counts), buffer)

    -- log(1 - input) * ((1 - target) / (N - counts[class])
    buffer:mul(input, -1):add(1):add(eps):log()
    output = output + torch.dot(buffer, torch.mul(target, -1):add(1):cdiv(NNK))

    if self.sizeAverage then
        output = output / input:nElement()
    end

    self.output = -output

    return self.output
end

function WeightedMultiLabelBinaryCrossEntropyCriterion:updateGradInput(input, target)
    -- - target / (counts[class] * input) + (1-target) / ((N-counts[class]) * (1-input))

    assert( input:nElement() == target:nElement(),
    "input and target size mismatch")

    self.buffer = self.buffer or input.new()

    local buffer = self.buffer
    local gradInput = self.gradInput

    if self.counts ~= nil and target:dim() ~= 1 then
        counts = self.counts:view(1, target:size(2)):expandAs(target)
    end
    if self.NNK ~= nil and target:dim() ~= 1 then
        NNK = self.NNK:view(1, target:size(2)):expandAs(target)
    end

    gradInput:resizeAs(input)
    -- - target / (counts[class] * input)
    gradInput:cdiv(target, torch.cmul(counts, input):add(eps)):mul(-1)

    buffer:resizeAs(input)
    -- (1-target) / ((N-counts[class]) * (1-input))
    buffer:mul(target, -1):add(1):cdiv(torch.cmul(NNK, torch.mul(input, -1):add(1)):add(eps))

    -- (- target / (counts[class] * input))  +  ((1-target) / ((N-counts[class]) * (input-1)))
    gradInput:add(buffer)

    if self.sizeAverage then
        gradInput:div(target:nElement())
    end

    return gradInput
end
