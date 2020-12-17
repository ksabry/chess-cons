#include <iostream>
#include "tournament-set.h"

TournamentSet::TournamentSet(std::mt19937 & randomEngine, ConsPlayer const * const * generation)
{
	ClearTournaments();

	// initialize how many tournaments each player has left to be assigned to
	int_fast32_t remainingPlayerTournaments[ConsPlayerConstants::generationSize];
	for (int_fast32_t playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
	{
		remainingPlayerTournaments[playerIndex] = ConsPlayerConstants::generationTournamentSize;
	}

	for (int_fast32_t tournamentIndex = 0; tournamentIndex < ConsPlayerConstants::generationSize; tournamentIndex++)
	{
		int_fast32_t remainingTournamentCount = ConsPlayerConstants::generationSize - tournamentIndex;
		for (int_fast32_t tournamentPlayerIndex = 0; tournamentPlayerIndex < ConsPlayerConstants::generationTournamentSize; tournamentPlayerIndex++)
		{
			// Find any players which have remainingTournamentCount assignments left and force that player to be part of this tournament
			// This is to ensure the constraint that each player may only be added to a specific tournament once will always possible
			bool found = false;
			for (int_fast32_t playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
			{
				if (remainingPlayerTournaments[playerIndex] == remainingTournamentCount)
				{
					AddPlayerToTournament(tournamentIndex, generation[playerIndex], playerIndex);
					remainingPlayerTournaments[playerIndex]--;
					found = true;
					break;
				}
			}
			if (found)
			{
				continue;
			}

			// Calculate size of pool of possible remaining players
			int_fast32_t playerPoolSize = 0;
			for (int_fast32_t playerIndex = 0; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
			{
				if (!IsPlayerInTournament(tournamentIndex, generation[playerIndex]))
				{
					playerPoolSize += remainingPlayerTournaments[playerIndex];
				}
			}

			std::uniform_int_distribution<int_fast32_t> selectionDistribution(0, playerPoolSize - 1);
			int_fast32_t selectionIndex = selectionDistribution(randomEngine);
			
			// Find the index of the player from the selection index; player index selection is weighted by how many tournament assignments they have remaining
			int_fast32_t playerIndex = 0;
			for (; playerIndex < ConsPlayerConstants::generationSize; playerIndex++)
			{
				if (!IsPlayerInTournament(tournamentIndex, generation[playerIndex]))
				{
					if (selectionIndex < remainingPlayerTournaments[playerIndex])
					{
						AddPlayerToTournament(tournamentIndex, generation[playerIndex], playerIndex);
						remainingPlayerTournaments[playerIndex]--;
						break;
					}
					else
					{
						selectionIndex -= remainingPlayerTournaments[playerIndex];
					}
				}
			}
		}
	}
}

void TournamentSet::ClearTournaments()
{
	for (int_fast32_t tournamentIndex = 0; tournamentIndex < ConsPlayerConstants::generationSize; tournamentIndex++)
	{
		for (int_fast32_t tournamentPlayerIndex = 0; tournamentPlayerIndex < ConsPlayerConstants::generationTournamentSize; tournamentPlayerIndex++)
		{
			tournaments[tournamentIndex][tournamentPlayerIndex] = nullptr;
			tournamentsByIndex[tournamentIndex][tournamentPlayerIndex] = -1;
		}
	}
}

void TournamentSet::AddPlayerToTournament(int_fast32_t tournamentIndex, ConsPlayer const * player, int_fast32_t playerIndex)
{
	// Very simple O(n) method
	for (int_fast32_t tournamentPlayerIndex = 0; tournamentPlayerIndex < ConsPlayerConstants::generationTournamentSize; tournamentPlayerIndex++)
	{
		if (tournaments[tournamentIndex][tournamentPlayerIndex] == nullptr)
		{
			tournaments[tournamentIndex][tournamentPlayerIndex] = player;
			tournamentsByIndex[tournamentIndex][tournamentPlayerIndex] = playerIndex;
			break;
		}
	}
}

bool TournamentSet::IsPlayerInTournament(int_fast32_t tournamentIndex, ConsPlayer const * player) const
{
	// Very simple O(n) method
	for (int_fast32_t tournamentPlayerIndex = 0; tournamentPlayerIndex < ConsPlayerConstants::generationTournamentSize; tournamentPlayerIndex++)
	{
		if (player == tournaments[tournamentIndex][tournamentPlayerIndex])
		{
			return true;
		}
	}
	return false;
}

std::ostream& operator <<(std::ostream& os, TournamentSet const & tournamentSet)
{
	os << "TournamentSet [" << std::endl;
	for (int_fast32_t tournamentIndex = 0; tournamentIndex < ConsPlayerConstants::generationSize; tournamentIndex++)
	{
		os << "  ";
		for (int_fast32_t tournamentPlayerIndex = 0; tournamentPlayerIndex < ConsPlayerConstants::generationTournamentSize; tournamentPlayerIndex++)
		{
			os << tournamentSet.tournamentsByIndex[tournamentIndex][tournamentPlayerIndex] << " ";
		}
		os << std::endl;
	}
	os << "]" << std::endl;
	return os;
}
