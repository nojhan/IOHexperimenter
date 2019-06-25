/** 
  (1+(lambda,lambda)) EA.
 **/


/**
 * The maximal budget for evaluations done by an optimization algorithm equals dimension * BUDGET_MULTIPLIER.
 * Increase the budget multiplier value gradually to see how it affects the runtime.
 */
static const size_t BUDGET_MULTIPLIER = 5;

/**
 * The maximal number of independent restarts allowed for an algorithm that restarts itself.
 */
static const size_t INDEPENDENT_RESTARTS = 11;

/**
 * The random seed. Change it if needed.
 */
static const uint32_t RANDOM_SEED = 1;


void generatingIndividual(int * individuals,
                            const size_t dimension, 
                            IOHprofiler_random_state_t *random_generator){
  size_t i;
  for(i = 0; i < dimension; ++i){
    individuals[i] = (int)(IOHprofiler_random_uniform(random_generator) * 2);
  }
}

void CopyIndividual(int * old, int * new, const size_t dimension){
  size_t i;
  for(i = 0; i < dimension; ++i){
    new[i] = old[i];
  }
}

void sampleNFromM(int* flip, size_t n, size_t m,IOHprofiler_random_state_t *random_generator)
{
  size_t i = 0;
  size_t j = 0;
  int temp,randPos;
  int * population = IOHprofiler_allocate_vector(m);
  for(i = 0; i < m; ++i){
    population[i] = i;
  }
  for(i = m-1; i > 0; --i){
    randPos = (int)(IOHprofiler_random_uniform(random_generator) * (i+1));
    temp = population[i];
    population[i] = population[randPos];
    population[randPos] = temp;
    flip[m-i-1] = population[i];
    if(m-i-1 == n-1){
      break;
    }
  }
  if(n == m){
    flip[n-1] = population[0];
  }
  IOHprofiler_free_memory(population);
}

/**
 * Binomial
 */
size_t randomBinomial(size_t n, double  probability,IOHprofiler_random_state_t *random_generator)
{
    size_t r, i;
    r = 0;
    for(i = 0; i < n; ++i){
        if(IOHprofiler_random_uniform(random_generator) < probability)
        {
            ++r;
        }
    }
    return r;
}

size_t crossover(int * individual, int * x, int * x_prime, size_t dimension, double crossover_rate, IOHprofiler_random_state_t *random_generator){
  size_t i,h,l;
  int flag,temp;
  int * flip;
  size_t ifDiff;
  ifDiff = 0;
  for(i = 0; i < dimension; ++i){
    individual[i] = x[i];
  }
  l = randomBinomial(dimension,crossover_rate,random_generator);
  if(l == 0 || l== dimension){
    return 0;
  }
  else{
    flip = IOHprofiler_allocate_int_vector(l);
    sampleNFromM(flip,l,dimension,random_generator);
    for(i = 0; i < l; ++i){
      if(individual[flip[i]] != x_prime[flip[i]]){
        individual[flip[i]] =  x_prime[flip[i]];
        ifDiff = 1;
      }
    }
  }
  IOHprofiler_free_memory(flip);
  if(ifDiff == 0){
    return 0;
  }
  return l;
}


size_t mutateIndividual(int * individual, 
                      const size_t dimension, 
                      double mutation_rate, 
                      IOHprofiler_random_state_t *random_generator){
  size_t i,h, l;
  int flag,temp;
  int * flip;

  l = mutation_rate * dimension;
  
  flip = IOHprofiler_allocate_int_vector(l);
  sampleNFromM(flip,l,dimension,random_generator);

  for(i = 0; i < l; ++i){
  individual[flip[i]] =  ((int)(individual[flip[i]] + 1) % 2);
  }
  IOHprofiler_free_memory(flip);
  return l;
}

int compareIndividuals(int * individual1, int * individual2, size_t dimension){
  size_t i;
  for(i = 0; i < dimension; ++i){
    if(individual1[i] != individual2[i])
      return 0;
  }
  return 1;
}

/**
 * An user defined algorithm.
 *
 * @param "evaluate" The function for evaluating variables' fitness. Invoking the 
 *        statement "evaluate(x,y)", then the fitness of 'x' will be stored in 'y[0]'.
 * @param "dimension" The dimension of problem.
 * @param "number_of_objectives" The number of objectives. The default is 1.
 * @param "lower_bounds" The lower bounds of the region of interested (a vector containing dimension values). 
 * @param "upper_bounds" The upper bounds of the region of interested (a vector containing dimension values). 
 * @param "max_budget" The maximal number of evaluations. You can set it by BUDGET_MULTIPLIER in "config" file.
 * @param "random_generator" Pointer to a random number generator able to produce uniformly and normally
 * distributed random numbers. You can set it by RANDOM_SEED in "config" file
 */
void User_Algorithm(evaluate_function_t evaluate,
                      const size_t dimension,
                      const size_t number_of_objectives,
                      const int *lower_bounds,
                      const int *upper_bounds,
                      const size_t max_budget,
                      IOHprofiler_random_state_t *random_generator) {

  /**
   * Add your algorithm in this function. You can invoke other self-defined functions,
   * but please remember this is the interface for IOHprofiler. Make sure your main
   * algorithm be inclueded in this function.
   *
   * The data of varibales and fitness will be stored once "evaluate()" works.
   *
   * If you want to store information of some self-defined parameters, use the statement
   * "set_parameters(size_t number_of_parameters,double *parameters)". The name of parameters
   * can be set in "config" file.
   */

  int *parent = IOHprofiler_allocate_int_vector(dimension);
  int *offspring = IOHprofiler_allocate_int_vector(dimension);
  int *offspring_prime = IOHprofiler_allocate_int_vector(dimension);
  int *best = IOHprofiler_allocate_int_vector(dimension);
  double best_value,a,b, best_value_mutation;
  double *y = IOHprofiler_allocate_vector(number_of_objectives);
  size_t number_of_parameters = 3;
  double *p = IOHprofiler_allocate_vector(number_of_parameters);
  size_t i, j,h,l,greedystep;
  int hit_optimal = 0, update_lambda_flag = 0;
  double lambda = 1.0;
  double mutation_rate = 1/(double)dimension,crossover_rate;
  l = 0;
  generatingIndividual(parent,dimension,random_generator);
  p[0] = lambda; p[1] = 0.0; p[2] = 0.0;
  set_parameters(number_of_parameters,p);
  evaluate(parent,y);
  CopyIndividual(parent,best,dimension);
  best_value = y[0];
  a = pow(1.5,0.25);
  b = 2.0/3.0;

  for (i = 1; i < max_budget; ) {
    update_lambda_flag = 0;
    mutation_rate = lambda/(double)dimension;
    crossover_rate = 1.0/lambda;


    l = randomBinomial(dimension,mutation_rate,random_generator);
    while(l == 0){
      l = randomBinomial(dimension,mutation_rate,random_generator);
    }

    best_value_mutation = -DBL_MAX;
    for(j = 0; j < (int)(lambda+0.5); ++j){
      CopyIndividual(parent,offspring,dimension);

      mutation_rate = l/(double)dimension;
      l = mutateIndividual(offspring,dimension,mutation_rate,random_generator);
      p[0] = lambda; p[1] = mutation_rate; p[2] = (double)l;
      /* Call the evaluate function to evaluate x on the current problem (this is where all the IOHprofiler logging
       * is performed) */
      set_parameters(number_of_parameters,p);
      evaluate(offspring, y);
      ++i;

      if(i == max_budget) break;
      if(if_hit_optimal()) {
        hit_optimal = 1;
        break;
      }


      if(y[0] > best_value_mutation){
        best_value_mutation = y[0];
        CopyIndividual(offspring,offspring_prime,dimension); 
      }

      if(y[0] > best_value){
        update_lambda_flag = 1;
      }

      if(y[0] >= best_value){
        best_value = y[0];
        CopyIndividual(offspring,best,dimension);
      }
    }

    if(i == max_budget) break;
    if(hit_optimal == 1) break;
    for(j = 0; j < (int)(lambda+0.5); ++j){
      l=crossover(offspring,parent,offspring_prime,dimension,crossover_rate,random_generator);
      if(l != 0){
        if(compareIndividuals(offspring,offspring_prime,dimension) == 1){
          continue;
        }
        p[0] = lambda; p[1] = crossover_rate + 1; p[2] = (double)l;
        evaluate(offspring,y);
        ++i;

        if(i == max_budget) break;
        if(if_hit_optimal()) {
          hit_optimal = 1;
          break;
        }

        if(y[0] > best_value){
          update_lambda_flag = 1;
        }

        if(y[0] >= best_value){
          best_value = y[0];
          CopyIndividual(offspring,best,dimension);
        }
      }
      else{
        continue;
      }
    }
  
    if(i == max_budget) break;
    if(hit_optimal == 1) break;
    
    if(update_lambda_flag == 1){
      lambda = (lambda * b) > 1 ? (lambda * b) : 1;
    }
    else{
      lambda =  (lambda * a) < (dimension) ? (lambda * a) : (dimension);
    }

    CopyIndividual(best,parent,dimension);
  }

  IOHprofiler_free_memory(parent);
  IOHprofiler_free_memory(offspring);
  IOHprofiler_free_memory(offspring_prime);
  IOHprofiler_free_memory(best);
  IOHprofiler_free_memory(p);
  IOHprofiler_free_memory(y);
}