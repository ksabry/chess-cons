#include <iostream>
#include "trainer.h"

int main()
{
	try
	{
		Trainer trainer;
		trainer.Train();
	}
	catch(const std::exception& e)
	{
		std::cerr << e.what() << '\n';
	}
}
