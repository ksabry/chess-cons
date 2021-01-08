#pragma once

#include "cons-player.h"

class TournamentSet
{
public:
	TournamentSet(std::mt19937 & randomEngine, ConsPlayer const * const * generation);
	ConsPlayer const * tournaments[ConsPlayerConstants::tournamentCount][ConsPlayerConstants::generationTournamentSize];
	int_fast32_t tournamentsByIndex[ConsPlayerConstants::tournamentCount][ConsPlayerConstants::generationTournamentSize];

private:
	void ClearTournaments();
	void AddPlayerToTournament(int_fast32_t tournamentIndex, ConsPlayer const * player, int_fast32_t playerIndex);
	bool IsPlayerInTournament(int_fast32_t tournamentIndex, ConsPlayer const * player) const;
};

std::ostream& operator <<(std::ostream& os, TournamentSet const & tournamentSet);
