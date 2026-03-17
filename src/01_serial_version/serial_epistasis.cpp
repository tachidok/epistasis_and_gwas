#include <iostream>
#include <fstream>
#include <vector>
#include <sstream>
#include <cmath>
#include <iomanip>
#include <string>

using namespace std;

// Calculates the Pearson correlation for the interaction of two SNPs
double calculate_pearson(const vector<unsigned int>& snp1, const vector<unsigned int>& snp2, const vector<unsigned int>& phenotype)
{
 // The size of the phenotype, number of samples
 int M = phenotype.size();
 
 double sum_X = 0, sum_Y = 0, sum_X2 = 0, sum_Y2 = 0, sum_XY = 0;

 // Loop over all the samples
 for (unsigned int i = 0; i < M; ++i)
  {
  // The epistatic interaction is the product of the two SNP values
  double X = snp1[i] * snp2[i];
  double Y = phenotype[i];
  
  sum_X += X;
  sum_Y += Y;
  sum_X2 += X * X;
  sum_Y2 += Y * Y;
  sum_XY += X * Y;
 }

 // Pearson correlation
 double numerator = (M * sum_XY) - (sum_X * sum_Y);
 double denominator_sq = ((M * sum_X2) - (sum_X * sum_X)) * ((M * sum_Y2) - (sum_Y * sum_Y));
 
 // Handle edge cases like zero variance to avoid NaN results
 if (denominator_sq <= 0.0) return 0.0; 
 
 return numerator / sqrt(denominator_sq);
 
}

int main(int argc, const char **argv)
{
 const unsigned int num_patients = 1000;
 const unsigned int num_snps = 5000;

 // Temporal variables to store the reading values from files
 string line, val, id;
 
 // 1. Read Phenotypes
 vector<unsigned int> phenotypes(num_patients);
 ifstream pheno_file("phenotypes.csv");

 // Check for error at opening the file
 if (!pheno_file.is_open())
  {
   cerr << "Error opening phenotypes.csv" << endl;
   return 1;
  }

 // Skip the first line
 getline(pheno_file, line);

 // Load data into memory
 for (unsigned int i = 0; i < num_patients; ++i)
  {
   // Get a line and parse it
   getline(pheno_file, line);
   stringstream ss(line);

   getline(ss, id, ','); // Ignore the identifier
   getline(ss, val); // Get the actual binary phenotype
   
   phenotypes.push_back(stoi(val));
  }
 pheno_file.close();
 
 // 2. Read Genotypes
 // We store as snps[snp_index][patient_index] for easier access later
 vector<vector<unsigned int> > snps(num_snps, vector<unsigned int>(num_patients));
 ifstream geno_file("genotypes.csv");
 
 // Check for errors at opening the files
 if (!geno_file.is_open()) {
  cerr << "Error opening genotypes.csv" << endl;
  return 1;
 }
 
 // Skip the first line
 getline(geno_file, line);
 
 // Load data into memory
 for (unsigned int i = 0; i < num_patients; ++i)
  {
   // Read a line
   getline(geno_file, line);
   stringstream ss(line);
   
   getline(ss, id, ','); // Ignore the identifier
   
   for (unsigned int s = 0; s < num_snps; ++s)
    {
     // Parse the reading
     getline(ss, val, ',');
     snps[s][i] = stoi(val);
    }
   
  }
 geno_file.close();
 
 // 3. Compute Epistatic Interactions (All unique pairs)
 cout << "--- Epistasis Interaction Scores ---" << endl;
 cout << fixed << setprecision(5);

 long long total_pairs = 0;
 unsigned int print_limit = 15; // Limit output to avoid flooding the console
 
 // N*(N-1)/2 comparisons
 for (unsigned int i = 0; i < num_snps; ++i)
  {
  for (unsigned int j = i + 1; j < num_snps; ++j)
   {
    double r = calculate_pearson(snps[i], snps[j], phenotypes);
    total_pairs++;
    
    if (total_pairs <= print_limit)
     {
      cout << "Pair (rs" << i << ", rs" << j << "): r = " << r << endl;
     }
    else if (total_pairs == print_limit + 1)
     {
      cout << "... continuing computations ..." << endl;
     }
   }
  
  }
 
 cout << "\nSuccessfully computed " << total_pairs << " interaction pairs." << endl;
 
 return 0;

}
