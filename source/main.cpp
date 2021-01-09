#include <iostream>
#include "trainer.h"
#include "tester.h"

int main()
{
	try
	{
		// Trainer trainer;
		// trainer.Train();
		Tester tester;
		tester.RoundRobin(40, 1);
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
}
