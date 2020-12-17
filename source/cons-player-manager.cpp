#include <queue>
#include <vector>
#include <cstdint>
#include <sstream>
#include "cons-player-manager.h"
#include "cons-player-constants.h"

#define FatalError(s) do {                                             \
	std::stringstream _where, _message;                                \
	_where << __FILE__ << ':' << __LINE__;                             \
	_message << std::string(s) + "\n" << __FILE__ << ':' << __LINE__;  \
	std::cerr << _message.str() << "\nAborting...\n";                  \
	cudaDeviceReset();                                                 \
	exit(1);                                                           \
} while(0)

#define checkCudnn(invocation) do {                                    \
	cudnnStatus_t status = invocation;                                 \
	std::stringstream _error;                                          \
	if (status != CUDNN_STATUS_SUCCESS) {                              \
		_error << "Cudnn failure: " << cudnnGetErrorString(status);    \
		FatalError(_error.str());                                      \
	}                                                                  \
} while(0)

#define checkCublas(invocation) do {                                   \
	cublasStatus_t status = invocation;                                \
	std::stringstream _error;                                          \
	if (status != CUBLAS_STATUS_SUCCESS) {                             \
		_error << "Cublas failure: " << cublasGetErrorString(status);  \
		FatalError(_error.str());                                      \
	}                                                                  \
} while(0)

#define checkCuda(invocation) do {                                     \
	cudaError_t status = invocation;                                   \
	std::stringstream _error;                                          \
	if (status != cudaSuccess) {                                       \
		_error << "Cuda failure: " << status;                          \
		FatalError(_error.str());                                      \
	}                                                                  \
} while(0)

char const * cublasGetErrorString(cublasStatus_t status)
{
	switch (status)
	{
		case CUBLAS_STATUS_SUCCESS:
			return "CUBLAS_STATUS_SUCCESS";
		case CUBLAS_STATUS_NOT_INITIALIZED:
			return "CUBLAS_STATUS_NOT_INITIALIZED";
		case CUBLAS_STATUS_ALLOC_FAILED:
			return "CUBLAS_STATUS_ALLOC_FAILED";
		case CUBLAS_STATUS_INVALID_VALUE:
			return "CUBLAS_STATUS_INVALID_VALUE";
		case CUBLAS_STATUS_ARCH_MISMATCH:
			return "CUBLAS_STATUS_ARCH_MISMATCH";
		case CUBLAS_STATUS_MAPPING_ERROR:
			return "CUBLAS_STATUS_MAPPING_ERROR";
		case CUBLAS_STATUS_EXECUTION_FAILED:
			return "CUBLAS_STATUS_EXECUTION_FAILED";
		case CUBLAS_STATUS_INTERNAL_ERROR:
			return "CUBLAS_STATUS_INTERNAL_ERROR";
	}
	return "<unknown>";
}

ConsPlayerManager::ConsPlayerManager()
{
	cudaDeviceProp deviceProp;
	checkCuda(cudaGetDeviceProperties(&deviceProp, 0));
	std::cout << "Using device \"" << deviceProp.name << "\"" << std::endl;

	checkCuda(cudaSetDevice(0));
	checkCublas(cublasCreate_v2(&cublasHandle));
	checkCudnn(cudnnCreate(&cudnnHandle));

	checkCudnn(cudnnCreateActivationDescriptor(&reluActivation));
	checkCudnn(cudnnSetActivationDescriptor(reluActivation, CUDNN_ACTIVATION_RELU, CUDNN_NOT_PROPAGATE_NAN, 10000000.0f));

	checkCuda(cudaStreamCreate(&tvStream));
	checkCuda(cudaStreamCreate(&cvStream));

	checkCuda(cudaMalloc(&d_tvcvInput, sizeof(float) * (ConsPlayerConstants::tvInputSize + ConsPlayerConstants::cvInputSize)));

	d_tvInput = d_tvcvInput;
	d_cvInput = d_tvcvInput + ConsPlayerConstants::tvInputSize;

	checkCuda(cudaMalloc(&d_cvOutput, sizeof(float) * 1));
	for (int_fast32_t layerIndex = 0; layerIndex < ConsPlayerConstants::cvLayerCount; layerIndex++)
	{
		checkCuda(cudaMalloc(&d_cvLayers[layerIndex], sizeof(float) * ConsPlayerConstants::cvLayerSizes[layerIndex]));
	}
	checkCuda(cudaMalloc(&d_tvOutput, sizeof(float) * 1));
	for (int_fast32_t layerIndex = 0; layerIndex < ConsPlayerConstants::tvLayerCount; layerIndex++)
	{
		checkCuda(cudaMalloc(&d_tvLayers[layerIndex], sizeof(float) * ConsPlayerConstants::tvLayerSizes[layerIndex]));
	}

	checkCuda(cudaMalloc(&d_whitePlayerTVWeights, sizeof(float) * ConsPlayerConstants::TVWeightCount()));
	checkCuda(cudaMalloc(&d_whitePlayerTVBiases, sizeof(float) * ConsPlayerConstants::TVBiasCount()));
	checkCuda(cudaMalloc(&d_whitePlayerCVWeights, sizeof(float) * ConsPlayerConstants::CVWeightCount()));
	checkCuda(cudaMalloc(&d_whitePlayerCVBiases, sizeof(float) * ConsPlayerConstants::CVBiasCount()));
	checkCuda(cudaMalloc(&d_blackPlayerTVWeights, sizeof(float) * ConsPlayerConstants::TVWeightCount()));
	checkCuda(cudaMalloc(&d_blackPlayerTVBiases, sizeof(float) * ConsPlayerConstants::TVBiasCount()));
	checkCuda(cudaMalloc(&d_blackPlayerCVWeights, sizeof(float) * ConsPlayerConstants::CVWeightCount()));
	checkCuda(cudaMalloc(&d_blackPlayerCVBiases, sizeof(float) * ConsPlayerConstants::CVBiasCount()));
}

ConsPlayerManager::~ConsPlayerManager()
{
	checkCuda(cudaFree(d_tvcvInput));
	checkCuda(cudaFree(d_cvOutput));
	for (int_fast32_t layerIndex = 0; layerIndex < ConsPlayerConstants::cvLayerCount; layerIndex++)
	{
		checkCuda(cudaFree(d_cvLayers[layerIndex]));
	}
	checkCuda(cudaFree(d_tvOutput));
	for (int_fast32_t layerIndex = 0; layerIndex < ConsPlayerConstants::tvLayerCount; layerIndex++)
	{
		checkCuda(cudaFree(d_tvLayers[layerIndex]));
	}

	checkCuda(cudaFree(d_whitePlayerTVWeights));
	checkCuda(cudaFree(d_whitePlayerTVBiases));
	checkCuda(cudaFree(d_whitePlayerCVWeights));
	checkCuda(cudaFree(d_whitePlayerCVBiases));
	checkCuda(cudaFree(d_blackPlayerTVWeights));
	checkCuda(cudaFree(d_blackPlayerTVBiases));
	checkCuda(cudaFree(d_blackPlayerCVWeights));
	checkCuda(cudaFree(d_blackPlayerCVBiases));

	checkCuda(cudaStreamDestroy(tvStream));
	checkCuda(cudaStreamDestroy(cvStream));

	checkCudnn(cudnnDestroyActivationDescriptor(reluActivation));
	checkCudnn(cudnnDestroy(cudnnHandle));
	checkCublas(cublasDestroy_v2(cublasHandle));
}

float ConsPlayerManager::PlayGame(ConsPlayer const * whitePlayer, ConsPlayer const * blackPlayer)
{
	this->SetupPlayers(whitePlayer, blackPlayer);

	GameState gameState;
	for (int_fast32_t moveIndex = 0; moveIndex < ConsPlayerConstants::maxGameMoveCount; moveIndex++)
	{
		GameState nextGameState;
		bool complete = this->PlayMove(gameState, &nextGameState);
		
		// std::cout << gameState;
		// std::cin.ignore();
		
		if (complete)
		{
			if (gameState.IsWhiteInCheck())
			{
				return -1.0f;
			}
			else if (gameState.IsBlackInCheck())
			{
				return 1.0f;
			}
			else
			{
				return 0.0f;
			}
		}
		gameState = nextGameState;
	}

	// Too many moves (e.g. infinite loop of moves); stalemate or tie break by piece score
	// TODO: we could possibly detect infinite loops early (or provide the relevant info to the network)
	if (ConsPlayerConstants::tieBreakWithPieceScore)
	{
		// For context the starting value of both sides is 41.52
		// Note that the total value on the board can, in rare cases due to promotion, but significantly higher
		float pieceScore = (gameState.WhitePieceScore() - gameState.BlackPieceScore()) / 50.0f;
		if (pieceScore < -0.9f)
		{
			pieceScore = -0.9f;
		}
		if (pieceScore > 0.9f)
		{
			pieceScore = 0.9f;
		}
		return pieceScore;
	}
	else
	{
		return 0.0f;
	}
}

bool ConsPlayerManager::PlayMove(GameState const & gameState, GameState * nextGameState)
{
	Tile playerColor = gameState.move;

	// TODO: when adding to unevaluatedNodes and another entry has the exact same score, it is likely the same state (reached by a different sequence of moves)
	//       in this case this is plenty of opportunity for optimization

	int_fast32_t nodeIndex = 0;
	int_fast32_t rootNodeCount = 0;

	// state node cv, state node index
	std::priority_queue<std::pair<float, int_fast32_t>, std::vector<std::pair<float, int_fast32_t>>, std::greater<std::pair<float, int_fast32_t>>> unevaluatedNodes;
	
	std::vector<GameState> nextGameStates;
	gameState.NextGameStates(nextGameStates);
	rootNodeCount = nextGameStates.size();
	if (rootNodeCount == 0)
	{
		return true;
	}

	// I have encountered stack overflow due to this, we could possibly try just increasing the stack size but for now we'll just put it on the heap
	GameStateNode * nodes = new GameStateNode[ConsPlayerConstants::maxMoveEvaluationCount + 256];

	for (int_fast32_t rootNodeIndex = 0; rootNodeIndex < rootNodeCount; rootNodeIndex++)
	{
		GameState& rootState = nextGameStates[rootNodeIndex];
		float tv, cv;
		this->EvaluateGameState(rootState, gameState, playerColor, 1, &tv, &cv);

		nodes[nodeIndex] = GameStateNode
		{
			rootState, // state
			nodeIndex, // rootMoveIndex
			1, // depth
			tv, // tv
			cv, // cv
			-1, // parentIndex
			-1, // childrenCount
		};
		// only add states with positive cv
		if (cv > 0)
		{
			unevaluatedNodes.push(std::pair<float, int_fast32_t>(cv, nodeIndex));
		}
		nodeIndex++;
	}

	while (nodeIndex < ConsPlayerConstants::maxMoveEvaluationCount && unevaluatedNodes.size() > 0)
	{
		// node with the highest consideration value
		int_fast32_t currentNodeIndex = unevaluatedNodes.top().second;
		unevaluatedNodes.pop();
		GameStateNode& currentNode = nodes[currentNodeIndex];

		nextGameStates.clear();
		gameState.NextGameStates(nextGameStates);
		if (nextGameStates.size() == 0)
		{
			// game has finished; we can set the minimax terminal value directly
			if (currentNode.state.IsWhiteInCheck())
			{
				currentNode.tv = -std::numeric_limits<float>::infinity();
			}
			else if (currentNode.state.IsBlackInCheck())
			{
				currentNode.tv = std::numeric_limits<float>::infinity();
			}
			else
			{
				currentNode.tv = 0;
			}
			continue;
		}

		currentNode.childrenCount = nextGameStates.size();
		for (int_fast32_t nextStateIndex = 0; nextStateIndex < nextGameStates.size(); nextStateIndex++)
		{
			GameState& nextState = nextGameStates[nextStateIndex];
			float tv, cv;
			EvaluateGameState(nextState, currentNode.state, playerColor, currentNode.depth + 1, &tv, &cv);

			nodes[nodeIndex] =
			{
				nextState, // state
				currentNode.rootMoveIndex, // rootMoveIndex
				currentNode.depth + 1, // depth
				tv, // tv
				cv, // cv
				-1, // parentIndex
				-1, // childrenCount
			};
			// only add states with positive cv
			if (cv > 0)
			{
				unevaluatedNodes.push(std::pair<float, int_fast32_t>(cv, nodeIndex));
			}
			currentNode.childrenIndices[nextStateIndex] = nodeIndex;
			nodeIndex++;
		}
	}

	// minimax over visited nodes
	// TODO: we can be intelligent about collapsing nodes as they are evaluated rather than waiting until the end
	uint_fast32_t bestRootIndex = 0;
	if (playerColor == Tile::White)
	{
		// maximize over white's moves
		float maximumScore = -std::numeric_limits<float>::infinity();
		for (int_fast32_t rootNodeIndex = 0; rootNodeIndex < rootNodeCount; rootNodeIndex++)
		{
			float rootNodeScore = this->MinimaxScoreNode(nodes[rootNodeIndex], nodes);
			if (rootNodeScore > maximumScore)
			{
				maximumScore = rootNodeScore;
				bestRootIndex = rootNodeIndex;
			}
		}
	}
	else
	{
		// minimize over black's moves
		float minimumScore = std::numeric_limits<float>::infinity();
		for (int_fast32_t rootNodeIndex = 0; rootNodeIndex < rootNodeCount; rootNodeIndex++)
		{
			float rootNodeScore = this->MinimaxScoreNode(nodes[rootNodeIndex], nodes);
			if (rootNodeScore < minimumScore)
			{
				minimumScore = rootNodeScore;
				bestRootIndex = rootNodeIndex;
			}
		}
	}
	*nextGameState = nodes[bestRootIndex].state;

	delete[] nodes;
	return false;
}

float ConsPlayerManager::MinimaxScoreNode(GameStateNode const & node, GameStateNode const * nodes)
{
	if (node.childrenCount <= 0)
	{
		return node.tv;
	}
	if (node.state.move == Tile::White)
	{
		// maximize over white's moves
		float maximumScore = -std::numeric_limits<float>::infinity();
		for (int_fast32_t childIndex = 0; childIndex < node.childrenCount; childIndex++)
		{
			GameStateNode const & childNode = nodes[node.childrenIndices[childIndex]];
			float childScore = MinimaxScoreNode(childNode, nodes);
			if (childScore > maximumScore)
			{
				maximumScore = childScore;
			}
		}
		return maximumScore;
	}
	else
	{
		// minimize over black's moves
		float minimumScore = std::numeric_limits<float>::infinity();
		for (int_fast32_t childIndex = 0; childIndex < node.childrenCount; childIndex++)
		{
			GameStateNode const & childNode = nodes[node.childrenIndices[childIndex]];
			float childScore = MinimaxScoreNode(childNode, nodes);
			if (childScore < minimumScore)
			{
				minimumScore = childScore;
			}
		}
		return minimumScore;
	}
}

// Sets tv (terminal value) and cv (consideration value)
void ConsPlayerManager::EvaluateGameState(GameState const & state, GameState const & previousState, Tile playerColor, int_fast32_t depth, float * tv, float * cv)
{
	// for testing
	// *cv = 1.0f;
	// *tv = state.WhitePieceScore() - state.BlackPieceScore() + (rand() % 10000) / 10000f;

	SetDeviceInput(state, previousState, depth);
	if (playerColor == Tile::White)
	{
		EvaluateTV(d_whitePlayerTVWeights, d_whitePlayerTVBiases);
		EvaluateCV(d_whitePlayerCVWeights, d_whitePlayerCVBiases);
		checkCuda(cudaDeviceSynchronize());
	}
	else
	{
		EvaluateTV(d_blackPlayerTVWeights, d_blackPlayerTVBiases);
		EvaluateCV(d_blackPlayerCVWeights, d_blackPlayerCVBiases);
		checkCuda(cudaDeviceSynchronize());
	}
	GetDeviceOutput(tv, cv);
}

void ConsPlayerManager::SetupPlayers(ConsPlayer const * whitePlayer, ConsPlayer const * blackPlayer)
{
	checkCuda(cudaMemcpy(
		d_whitePlayerTVWeights,
		whitePlayer->weights, sizeof(float) * ConsPlayerConstants::TVWeightCount(),
		cudaMemcpyHostToDevice
	));
	checkCuda(cudaMemcpy(
		d_whitePlayerTVBiases,
		whitePlayer->biases, sizeof(float) * ConsPlayerConstants::TVBiasCount(),
		cudaMemcpyHostToDevice
	));
	checkCuda(cudaMemcpy(
		d_whitePlayerCVWeights,
		whitePlayer->weights + ConsPlayerConstants::TVWeightCount(), sizeof(float) * ConsPlayerConstants::CVWeightCount(),
		cudaMemcpyHostToDevice
	));
	checkCuda(cudaMemcpy(
		d_whitePlayerCVBiases,
		whitePlayer->biases + ConsPlayerConstants::TVBiasCount(), sizeof(float) * ConsPlayerConstants::CVBiasCount(),
		cudaMemcpyHostToDevice
	));
	checkCuda(cudaMemcpy(
		d_blackPlayerTVWeights,
		blackPlayer->weights, sizeof(float) * ConsPlayerConstants::TVWeightCount(),
		cudaMemcpyHostToDevice
	));
	checkCuda(cudaMemcpy(
		d_blackPlayerTVBiases,
		blackPlayer->biases, sizeof(float) * ConsPlayerConstants::TVBiasCount(),
		cudaMemcpyHostToDevice
	));
	checkCuda(cudaMemcpy(
		d_blackPlayerCVWeights,
		blackPlayer->weights + ConsPlayerConstants::TVWeightCount(), sizeof(float) * ConsPlayerConstants::CVWeightCount(),
		cudaMemcpyHostToDevice
	));
	checkCuda(cudaMemcpy(
		d_blackPlayerCVBiases,
		blackPlayer->biases + ConsPlayerConstants::TVBiasCount(), sizeof(float) * ConsPlayerConstants::CVBiasCount(),
		cudaMemcpyHostToDevice
	));
}

void ConsPlayerManager::SetDeviceInput(GameState const & gameState, GameState const & previousGameState, int_fast32_t depth)
{
	// terminal value network input
	// 1-hot encoding of current board with piece color differentiated consistently as white/black in the normal manner
	float tvcvInput[ConsPlayerConstants::tvInputSize + ConsPlayerConstants::cvInputSize] = {};
	for (int_fast32_t y = 0; y < 8; y++)
	{
		for (int_fast32_t x = 0; x < 8; x++)
		{
			switch (gameState.GetTile(x, y))
			{
				case Tile::Empty:       tvcvInput[(y * 8 + x) * 13 +  0] = 1.0f; break;
				case Tile::WhitePawn:   tvcvInput[(y * 8 + x) * 13 +  1] = 1.0f; break;
				case Tile::WhiteRook:   tvcvInput[(y * 8 + x) * 13 +  2] = 1.0f; break;
				case Tile::WhiteKnight: tvcvInput[(y * 8 + x) * 13 +  3] = 1.0f; break;
				case Tile::WhiteBishop: tvcvInput[(y * 8 + x) * 13 +  4] = 1.0f; break;
				case Tile::WhiteQueen:  tvcvInput[(y * 8 + x) * 13 +  5] = 1.0f; break;
				case Tile::WhiteKing:   tvcvInput[(y * 8 + x) * 13 +  6] = 1.0f; break;
				case Tile::BlackPawn:   tvcvInput[(y * 8 + x) * 13 +  7] = 1.0f; break;
				case Tile::BlackRook:   tvcvInput[(y * 8 + x) * 13 +  8] = 1.0f; break;
				case Tile::BlackKnight: tvcvInput[(y * 8 + x) * 13 +  9] = 1.0f; break;
				case Tile::BlackBishop: tvcvInput[(y * 8 + x) * 13 + 10] = 1.0f; break;
				case Tile::BlackQueen:  tvcvInput[(y * 8 + x) * 13 + 11] = 1.0f; break;
				case Tile::BlackKing:   tvcvInput[(y * 8 + x) * 13 + 12] = 1.0f; break;
			}
		}
	}

	// consideration value network input
	// 1-hot encoding of both current and previous board with pieces differentiated consistently as current mover's color/opponent's color rather than white/black; also includes depth
	for (int s = 0; s < 2; s++)
	{
		for (int y = 0; y < 8; y++)
		{
			for (int x = 0; x < 8; x++)
			{
				GameState const & state = s == 0 ? gameState : previousGameState;
				bool invert = gameState.move == Tile::White;
				switch (state.GetTile(x, y))
				{
					case Tile::Empty:       tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13] = 1.0f; break;
					case Tile::WhitePawn:   tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  7 :  1)] = 1.0f; break;
					case Tile::WhiteRook:   tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  8 :  2)] = 1.0f; break;
					case Tile::WhiteKnight: tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  9 :  3)] = 1.0f; break;
					case Tile::WhiteBishop: tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ? 10 :  4)] = 1.0f; break;
					case Tile::WhiteQueen:  tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ? 11 :  5)] = 1.0f; break;
					case Tile::WhiteKing:   tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ? 12 :  6)] = 1.0f; break;
					case Tile::BlackPawn:   tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  1 :  7)] = 1.0f; break;
					case Tile::BlackRook:   tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  2 :  8)] = 1.0f; break;
					case Tile::BlackKnight: tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  3 :  9)] = 1.0f; break;
					case Tile::BlackBishop: tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  4 : 10)] = 1.0f; break;
					case Tile::BlackQueen:  tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  5 : 11)] = 1.0f; break;
					case Tile::BlackKing:   tvcvInput[ConsPlayerConstants::tvInputSize + ((s * 8 + y) * 8 + x) * 13 + (invert ?  6 : 12)] = 1.0f; break;
				}
			}
		}
	}
	tvcvInput[ConsPlayerConstants::tvInputSize + 2 * 64 * 13] = depth;

	checkCuda(cudaMemcpy(d_tvcvInput, tvcvInput, sizeof(float) * (ConsPlayerConstants::tvInputSize + ConsPlayerConstants::cvInputSize), cudaMemcpyHostToDevice));
}

void ConsPlayerManager::GetDeviceOutput(float * tv, float * cv)
{
	checkCuda(cudaMemcpy(tv, d_tvOutput, sizeof(float) * 1, cudaMemcpyDeviceToHost));
	checkCuda(cudaMemcpy(cv, d_cvOutput, sizeof(float) * 1, cudaMemcpyDeviceToHost));
}

// Assumes cvInput is already populated and writes to cvOutput
void ConsPlayerManager::EvaluateCV(float * weights, float * biases)
{
	checkCublas(cublasSetStream_v2(cublasHandle, cvStream));
	checkCudnn(cudnnSetStream(cudnnHandle, cvStream));

	int_fast32_t weightOffset = 0;
	int_fast32_t biasOffset = 0;

	for (int layerIndex = 0; layerIndex <= ConsPlayerConstants::cvLayerCount; layerIndex++)
	{
		int_fast32_t lastSize = layerIndex == 0 ? ConsPlayerConstants::cvInputSize : ConsPlayerConstants::cvLayerSizes[layerIndex - 1];
		int_fast32_t nextSize = layerIndex == ConsPlayerConstants::cvLayerCount ? 1 : ConsPlayerConstants::cvLayerSizes[layerIndex];

		float * lastLayer = layerIndex == 0 ? d_cvInput : d_cvLayers[layerIndex - 1];
		float * nextLayer = layerIndex == ConsPlayerConstants::cvLayerCount ? d_cvOutput : d_cvLayers[layerIndex];

		float * layerWeights = weights + weightOffset;
		float * layerBiases = biases + biasOffset;

		PropogateFeedForwardLayer(lastLayer, lastSize, nextLayer, nextSize, layerWeights, layerBiases, layerIndex != ConsPlayerConstants::cvLayerCount);

		weightOffset += lastSize * nextSize;
		biasOffset += nextSize;
	}
}

// Assumes tvInput is already populated and writes to tvOutput
void ConsPlayerManager::EvaluateTV(float * weights, float * biases)
{
	checkCublas(cublasSetStream_v2(cublasHandle, tvStream));
	checkCudnn(cudnnSetStream(cudnnHandle, tvStream));

	int_fast32_t weightOffset = 0;
	int_fast32_t biasOffset = 0;

	for (int layerIndex = 0; layerIndex <= ConsPlayerConstants::tvLayerCount; layerIndex++)
	{
		int_fast32_t lastSize = layerIndex == 0 ? ConsPlayerConstants::tvInputSize : ConsPlayerConstants::tvLayerSizes[layerIndex - 1];
		int_fast32_t nextSize = layerIndex == ConsPlayerConstants::tvLayerCount ? 1 : ConsPlayerConstants::tvLayerSizes[layerIndex];

		float * lastLayer = layerIndex == 0 ? d_tvInput : d_tvLayers[layerIndex - 1];
		float * nextLayer = layerIndex == ConsPlayerConstants::tvLayerCount ? d_tvOutput : d_tvLayers[layerIndex];

		float * layerWeights = weights + weightOffset;
		float * layerBiases = biases + biasOffset;

		PropogateFeedForwardLayer(lastLayer, lastSize, nextLayer, nextSize, layerWeights, layerBiases, layerIndex != ConsPlayerConstants::tvLayerCount);
		
		weightOffset += lastSize * nextSize;
		biasOffset += nextSize;
	}
}

void ConsPlayerManager::PropogateFeedForwardLayer(
	float * input,
	size_t inputSize,
	float * output,
	size_t outputSize,
	float * weights,
	float * biases,
	bool applyRelu
)
{
	float const zero = 0.0f;
	float const one = 1.0f;

	// output = weights * input
	checkCublas(cublasSgemv_v2(
		cublasHandle, /* handle */
		CUBLAS_OP_N,  /* trans  */
		inputSize,    /* m      */
		outputSize,   /* n      */
		&one,         /* alpha  */
		weights,      /* A      */
		inputSize,    /* lda    */
		input,        /* x      */
		1,            /* incx   */
		&zero,        /* beta   */
		output,       /* y      */
		1             /* incy   */
	));

	// output = output + biases
	checkCublas(cublasSaxpy_v2(
		cublasHandle, /* handle */
		outputSize,   /* n      */
		&one,         /* alpha  */
		biases,       /* x      */
		1,            /* incx   */
		output,       /* y      */
		1             /* incy   */
	));

	if (applyRelu)
	{
		// TODO: these should not be done here
		cudnnTensorDescriptor_t desc;
		checkCudnn(cudnnCreateTensorDescriptor(&desc));
		checkCudnn(cudnnSetTensor4dDescriptor(
			desc,
			CUDNN_TENSOR_NCHW,
			CUDNN_DATA_FLOAT,
			1, 1, outputSize, 1
		));
		// output = relu(output)
		checkCudnn(cudnnActivationForward(
			cudnnHandle,    /* handle         */
			reluActivation, /* activationDesc */
			&one,           /* alpha          */
			desc,           /* xDesc          */
			output,         /* x              */
			&zero,          /* beta           */
			desc,           /* yDesc          */
			output          /* y              */
		));
		checkCudnn(cudnnDestroyTensorDescriptor(desc));
	}
}