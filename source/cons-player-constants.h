#pragma once

#include <cstdint>

namespace ConsPlayerConstants
{
	constexpr int_fast32_t version = 1;
	constexpr uint_fast32_t seed = 123;

	constexpr int_fast32_t tvInputSize = 64 * 13;
	constexpr int_fast32_t cvInputSize = 2 * 64 * 13 + 1;
	constexpr int_fast32_t inputSize = tvInputSize + cvInputSize;

	// constexpr int_fast32_t tvLayerSizes[] = { 1024, 512, 256, 256, 256 };
	// constexpr int_fast32_t cvLayerSizes[] = { 1024, 512, 256, 256, 256 };
	constexpr int_fast32_t tvLayerSizes[] = { 256, 128 };
	constexpr int_fast32_t cvLayerSizes[] = { 256, 128 };
	constexpr int_fast32_t tvLayerCount = sizeof(tvLayerSizes) / sizeof(int_fast32_t);
	constexpr int_fast32_t cvLayerCount = sizeof(cvLayerSizes) / sizeof(int_fast32_t);

	constexpr int_fast32_t maxGameMoveCount = 100;
	constexpr int_fast32_t maxMoveEvaluationCountLow = 300;
	constexpr int_fast32_t maxMoveEvaluationCountHigh = 1000;
	constexpr bool tieBreakWithPieceScore = true;

	constexpr int_fast32_t generationSize = 30;
	constexpr int_fast32_t generationTournamentSize = 4;
	static_assert(generationSize % 2 == 0);
	static_assert(generationTournamentSize % 2 == 0);
	constexpr int_fast32_t tournamentCount = generationSize / 2;
	constexpr int_fast32_t playerTournamentCount = generationTournamentSize / 2;
	constexpr float mutationRate = 0.1;
	constexpr float mutationAmount = 0.1;

	constexpr int_fast32_t TVWeightCount()
	{
		int_fast32_t count = tvInputSize * tvLayerSizes[0];
		for (int_fast32_t layerIndex = 1; layerIndex < tvLayerCount; layerIndex++)
		{
			count += tvLayerSizes[layerIndex - 1] * tvLayerSizes[layerIndex];
		}
		count += tvLayerSizes[tvLayerCount - 1];
		return count;
	}

	constexpr int_fast32_t CVWeightCount()
	{
		int_fast32_t count = cvInputSize * cvLayerSizes[0];
		for (int_fast32_t layerIndex = 1; layerIndex < cvLayerCount; layerIndex++)
		{
			count += cvLayerSizes[layerIndex - 1] * cvLayerSizes[layerIndex];
		}
		count += cvLayerSizes[cvLayerCount - 1];
		return count;
	}

	constexpr int_fast32_t WeightCount()
	{
		return TVWeightCount() + CVWeightCount();
	}

	constexpr int_fast32_t TVBiasCount()
	{
		int_fast32_t count = 0;
		for (int_fast32_t layerIndex = 0; layerIndex < tvLayerCount; layerIndex++)
		{
			count += tvLayerSizes[layerIndex];
		}
		count += 1;
		return count;
	}

	constexpr int_fast32_t CVBiasCount()
	{
		int_fast32_t count = 0;
		for (int_fast32_t layerIndex = 0; layerIndex < cvLayerCount; layerIndex++)
		{
			count += cvLayerSizes[layerIndex];
		}
		count += 1;
		return count;
	}

	constexpr int_fast32_t BiasCount()
	{
		return TVBiasCount() + CVBiasCount();
	}
}
