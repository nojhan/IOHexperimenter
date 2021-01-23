#include <cmath>
#include <vector>
#include <string>
#include <algorithm>
#include <cassert>
#include <iostream>
#include <random>
#include <iterator>

#include <IOHprofiler_BBOB_suite.hpp>
#include <IOHprofiler_problem.h>
#include <IOHprofiler_ecdf_logger.h>

int main()
{
    size_t sample_size = 100;

    std::vector<int> pbs = {1,2};
    std::vector<int> ins = {1,2};
    std::vector<int> dims = {2,10};

    BBOB_suite bench(pbs,ins,dims);
    // bench.loadProblem();

    size_t ecdf_width = 20;
    using Logger = IOHprofiler_ecdf_logger<BBOB_suite::InputType>;
    IOHprofiler_RangeLog<double> error(0,6e7,ecdf_width);
    IOHprofiler_RangeLog<size_t> evals(0,sample_size,ecdf_width);
    Logger logger(error, evals);

    logger.activate_logger();
    logger.track_suite(bench);

    size_t seed = 5;
    std::mt19937 gen(seed);
    // std::mt19937 gen(time(0));
    std::uniform_real_distribution<> dis(-5,5);

    BBOB_suite::Problem_ptr pb;
    size_t n=0;
    while((pb = bench.get_next_problem())) {
        logger.track_problem(*pb);

        size_t d = pb->IOHprofiler_get_number_of_variables();
        for(size_t s=0; s < sample_size; ++s) {
            std::vector<double> sol;
            sol.reserve(d);
            std::generate_n(std::back_inserter(sol), d, [&dis,&gen](){return dis(gen);});

            double f = pb->evaluate(sol);
            logger.do_log(pb->loggerInfo());
        }

        n++;
    } // for name_id

    size_t i,j,k;
    std::tie(i,j,k) = logger.size();
    assert(i == pbs.size());
    assert(j == dims.size());
    assert(k == ins.size());

    for(int ipb : pbs) {
        for(int idim : dims) {
            for(int iins : ins) {
                const auto& m = logger.at(ipb, iins, idim);
                assert(m.size() == ecdf_width);
                assert(m[0].size() == ecdf_width);
            }
        }
    }

    IOHprofiler_ecdf_aggregate sum;
    IOHprofiler_ecdf_aggregate::Mat mat = sum(logger.data());
    std::clog << "Attainments aggregate: " << std::endl;
    assert(mat.size() > 0);
    assert(mat[0].size() > 1);
    for(long i = mat.size()-1; i >= 0; --i) {
        std::cout << mat[i][0];
        for(long j = 1; j < mat[i].size(); ++j) {
            std::cout << ", " << mat[i][j];
        }
        std::cout << std::endl;
    }
}
