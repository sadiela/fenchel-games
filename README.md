# fenchel-games
Final Project for EC500: Introduction to Online Learning

# Project Goals
1. Verify equivalence of several convex optimization algorithm and "fenchel games"
2. Try some new combinations of OL algorithms (or weights, etc.) to "discover" new convex optimization algorithms
3. Experts + teams

# Takeaways
1. Why do? You probably wouldn't.
    * Many of the algorithms require you to solve another optimization problem at each time step (intractable if that optimization problem does not have a closed form)
    * No reason to solve the Fenchel game over the corresponding convex optimization algorithm (unless conjugate is "simpler"?)
2. Good for deriving new algorithms & understanding the connections between seemingly unrelated convex optimization algorithms
3. Creating a **general library** is hard! Many function- and regularizer-dependent steps. 
