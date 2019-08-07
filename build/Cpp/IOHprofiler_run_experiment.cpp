#include "../../src/Template/Experiments/IOHprofiler_experimenter.hpp"

IOHprofiler_random random_generator(1);
static int budget_scale = 100;

std::vector<int> Initialization(int dimension) {
  std::vector<int> x;
  x.reserve(dimension);
  for (int i = 0; i != dimension; ++i) {
      x.push_back((int)(random_generator.IOHprofiler_uniform_rand() * 2));
  }
  return x;
};

int mutation(std::vector<int> &x, double mutation_rate) {
  int result = 0;
  int n = x.size();
  for(int i = 0; i != n; ++i) {
    if(random_generator.IOHprofiler_uniform_rand() < mutation_rate) {
      x[i] = (x[i] + 1) % 2;
      result = 1;
    }
  }
  return result;
}

/// This is an (1+1)_EA with static mutation rate = 1/n.
void evolutionary_algorithm(std::shared_ptr<IOHprofiler_problem<int>> problem, std::shared_ptr<IOHprofiler_csv_logger> logger) {
  /// Declaration for variables in the algorithm
  std::vector<int> x;
  std::vector<int> x_star;
  std::vector<double> y;
  double best_value;
  double mutation_rate = 1.0/problem->IOHprofiler_get_number_of_variables();
  int budget = budget_scale * problem->IOHprofiler_get_number_of_variables() * problem->IOHprofiler_get_number_of_variables();

  x = Initialization(problem->IOHprofiler_get_number_of_variables());
  copyVector(x,x_star);
  y = problem->evaluate(x);
  logger->write_line(problem->loggerInfo());
  best_value = y[0];

  int count = 0;
  while (count <= budget && !problem->IOHprofiler_hit_optimal()) {
    copyVector(x_star,x);
    if (mutation(x,mutation_rate)) {
      y = problem->evaluate(x);
      logger->write_line(problem->loggerInfo());
    }
    if (y[0] >= best_value) {
      best_value = y[0];
      copyVector(x,x_star);
    }
    count++;
  }
}

void _run_experiment() {
  std::string configName = "./configuration.ini";
  IOHprofiler_experimenter<int> experimenter(configName,evolutionary_algorithm);
  experimenter._set_independent_runs(2);
  experimenter._run();
}

int main(){
  _run_experiment();
  return 0;
}
