#pragma once

#include <cstdint>
#include <unordered_map>
#include <cuda_runtime.h>
#include <cublas_v2.h>
#include <cudnn.h>

#include "game-state.h"
#include "cons-player.h"

struct GameStateNode
{
	GameState state;
	int_fast32_t rootMoveIndex;
	int_fast32_t depth;
	// terminal value
	float tv;
	// consideration value
	float cv;
	int_fast32_t parentIndex;
	int_fast32_t childrenIndices[256];
	int_fast32_t childrenCount;
};

using EvaluationCache = std::unordered_map<GameStateSignature, std::pair<float, float>>;

class ConsPlayerManager
{
public:
	ConsPlayerManager();
	~ConsPlayerManager();

	float PlayGame(ConsPlayer const * whitePlayer, ConsPlayer const * blackPlayer, int_fast32_t maxMoveEvaluationCount, EvaluationCache * whiteCache = nullptr, EvaluationCache * blackCache = nullptr);
	// returns true if the game is complete (there are no valid moves), false otherwise
	bool PlayMove(GameState & gameState, GameState * nextGameState, int_fast32_t maxMoveEvaluationCount, EvaluationCache * playerCache = nullptr);
	float MinimaxScoreNode(GameStateNode const & node, GameStateNode const * nodes);
	void EvaluateGameState(GameState & state, GameState & previousState, Tile playerColor, int_fast32_t depth, float * tv, float * cv, EvaluationCache * playerCache = nullptr);

private:
	cublasHandle_t cublasHandle;
	cudnnHandle_t cudnnHandle;
	cudnnActivationDescriptor_t reluActivation;

	cudaStream_t tvStream;
	cudaStream_t cvStream;

	float * d_tvcvInput;
	float * d_tvInput;
	float * d_cvInput;
	float * d_cvOutput;
	float * d_cvLayers[ConsPlayerConstants::cvLayerCount];
	float * d_tvOutput;
	float * d_tvLayers[ConsPlayerConstants::tvLayerCount];

	float * d_whitePlayerTVWeights;
	float * d_whitePlayerTVBiases;
	float * d_whitePlayerCVWeights;
	float * d_whitePlayerCVBiases;
	float * d_blackPlayerTVWeights;
	float * d_blackPlayerTVBiases;
	float * d_blackPlayerCVWeights;
	float * d_blackPlayerCVBiases;

	void SetupPlayers(ConsPlayer const * whitePlayer, ConsPlayer const * blackPlayer);
	void SetDeviceInput(GameState const & gameState, GameState const & previousGameState, int_fast32_t depth);
	void GetDeviceOutput(float * tv, float * cv);

	void EvaluateCV(float * weights, float * biases);
	void EvaluateTV(float * weights, float * biases);
	void PropogateFeedForwardLayer(
		float * input,
		size_t inputSize,
		float * output,
		size_t outputSize,
		float * weights,
		float * biases,
		bool applyRelu
	);
};
