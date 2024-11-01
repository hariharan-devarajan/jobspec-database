using Cxx
using Libdl
using LinearAlgebra
using StatsBase
using Colors
using Random

using BSON: @save, @load
include("pancakerobot.jl")

const ID = parse(Int, ARGS[1])
const ARG_ENV = parse(Int, ARGS[2])
const GENERATIONS = 200
const RATE = [90/10, MAX_PRESS/10, 9/10]
const DIR = "genomes_closed_$(ARG_ENV)"

RNG = Random.MersenneTwister(ID*10000)

function wrap(array, low, high)
	arr = deepcopy(array)
	for i in 1:size(arr)[1]
		for j in 1:size(arr)[2]
			if arr[i, j] < low
				arr[i, j] = (high + arr[i, j]) + 1
				#arr[i, j] = low
			end
			if arr[i, j] > high
				arr[i, j] = (arr[i, j] - high) - 1
				#arr[i, j] = high
			end
		end
	end
	return Int64.(arr)
end

function new_genome()
	global RNG
	genome = nothing
	if ARG_ENV == 0
		genome = [0, MAX_PRESS, rand(RNG, 0:8, 10, 2)]
	elseif ARG_ENV == 15
		genome = [90, 0, rand(RNG, 0:8, 10, 2)]
	end
	return genome
end

function mutate_genome(genome)
	global RNG
	newg = deepcopy(genome)

	gene = round.(RATE[3] .* randn(RNG, 10, 2) .+ genome[3])
	gene = wrap(gene, 0, 8)
	newg[3] = gene
	return newg
end

results = nothing
gen = nothing
worker = nothing
fit = nothing

if isfile("$DIR/worker_$(lpad(ID, 2, "0"))_results.bson")
	global results, gen, worker, fit, RNG
	@load "$DIR/worker_$(lpad(ID, 2, "0"))_results.bson" results
	gen, worker, fit, RNG = results[end]
else
	global results, gen, worker, fit, RNG
	gen = 0
	worker = new_genome()
	fit, _ = fitness(worker, ARG_ENV)
	results = [(gen, worker, fit, deepcopy(RNG))]
	@save "$DIR/worker_$(lpad(ID, 2, "0"))_results.bson" results
	f = open("$DIR/worker_$(lpad(ID, 2, "0"))_best.txt", "w")
	write(f, "$fit")
	close(f)
end

for g in gen+1:GENERATIONS
	global results, worker, fit
	spawn = mutate_genome(worker)
	spawn_fit, _ = fitness(spawn, ARG_ENV)
	if spawn_fit > fit
		worker = spawn
		fit = spawn_fit
	end
	push!(results, (g, worker, fit, deepcopy(RNG)))
	@save "$DIR/worker_$(lpad(ID, 2, "0"))_results.bson" results
	f = open("$DIR/worker_$(lpad(ID, 2, "0"))_best.txt", "w")
	write(f, "$fit")
	close(f)
end