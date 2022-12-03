use indicatif::{ProgressBar, ProgressDrawTarget,ProgressStyle};
use nalgebra::{Point2, DMatrix, distance, partial_cmp};
use rand::{prelude::{SliceRandom, Distribution, ThreadRng}, distributions::Uniform};
use std::{collections::{LinkedList, HashMap}, cmp::Ordering};
use plotters::{prelude::{BitMapBackend, IntoDrawingArea, ChartBuilder, EmptyElement, Circle}, series::{PointSeries, LineSeries}};
use plotters::style::{BLUE, BLACK,WHITE,RED};
use rayon::prelude::*;

#[derive(Clone)]
pub struct Cell {
    solution: Vec<usize>,
    fitness: f32,
    nucleosome: Vec<bool>,
    father: Option<Vec<usize>>,
    mother:Option<Vec<usize>>,
}

pub enum Mechanisms {
    Imprinting(f32),
    Reprogramming(f32),
    Position(f32),
}

pub struct EpigeneticSearch {
    pub individual_nb: u16,
    pub cells_nb: u16,
    pub nucleo_prob: f32,
    pub nucleo_rad: i32,
    pub mechanisms: Vec<Mechanisms>,
    pub max_epochs: usize,
    pub distances: DMatrix<f32>,
    pub best_solution: Option<Vec<usize>>,
}

impl EpigeneticSearch {
    /// Initialises an instance of the EpigeneticSearch with its parameters.
    /// 
    /// # Arguments
    /// 
    /// * `individual_nb` - numbers of individuals.
    /// * `cells_nb` - number of cells per each individual
    /// * `nucleo_prob` - Probability of mutation in the nucleosome.
    /// * `nucleo_rad` - Radius of nucleosome when mutating.
    /// * `mechanisms` - List of `Mechanisms` to use in mutations
    /// * `max_epochs` - Maximum number of iterations to run
    /// 
    pub fn init(individual_nb: u16, cells_nb:u16,
        nucleo_prob:f32,
        nucleo_rad: i32, mechanisms:Vec<Mechanisms>,
        max_epochs: usize, best_sol:Option<Vec<usize>>) -> EpigeneticSearch {
            EpigeneticSearch { 
                individual_nb: individual_nb,
                cells_nb: cells_nb,
                nucleo_prob: nucleo_prob,
                nucleo_rad: nucleo_rad,
                mechanisms: mechanisms,
                max_epochs: max_epochs,
                distances: DMatrix::<f32>::zeros(1,1),
                best_solution:best_sol,
            }
        }

    /// Performs the algorithm for the given cartesian coordinates provided.
    /// 
    /// # Arguments
    /// 
    /// * `coordinates` - Vector of coordinates for the TSP problem.
    /// 
    pub fn call(&mut self, coordinates:Vec<Point2<f32>>, _optimum_path:Option<Vec<usize>>) -> Result<(), Box<dyn std::error::Error>>{
        let distances = calculate_dist(coordinates.clone());
        self.distances = distances;

        let mut population = self.init_population();
        
        let mut i: usize = 0;
        let mut fitnesses: LinkedList<f32> = LinkedList::new();

        let pb = ProgressBar::new(self.max_epochs as u64);
        pb.set_draw_target(ProgressDrawTarget::stdout());
        pb.set_style(ProgressStyle::with_template("[{prefix}] [{elapsed_precise}] {bar:40.cyan/blue} {pos:>7}/{len:7} {msg}").unwrap());
        pb.tick();

        while i< self.max_epochs { //TODO termination by stale fitness
            
            let mut newpop = population.clone();
            newpop = self.nucleosome_generation(newpop);
            newpop = self.nucleosome_reproduction(newpop);
            newpop = self.epigen_mechanism(newpop);
            
            // replace population for with the top best fitness individuals
            population.append(&mut newpop);
            population.par_sort_by(
                |a,b|
                evaluate_individual(a)
                .partial_cmp(&evaluate_individual(b))
                .unwrap_or(Ordering::Greater)
            );
            population = population[0..self.individual_nb as usize].to_vec();
            
            // add mean fitness to the linked list to see if the progression stales
            let avg_fitness: f32 = population
            .par_iter()
            .map(|x| evaluate_individual(x))
            .sum::<f32>() / population.len() as f32;
            pb.inc(1);
            pb.set_prefix(format!("Loss: {}",&avg_fitness));
            // println!("Pass {}, Loss: {}",i,&avg_fitness);
            fitnesses.push_front(avg_fitness);
            if fitnesses.len()>4{
                fitnesses.pop_back();
            }

            i+=1;
        }

        pb.finish_and_clear();

        let best_cell: Cell = select_best_cell(population);
        println!("Finished with Loss {}",&best_cell.fitness);

        if self.best_solution.is_some() {
            let opt_fitness = evaluate_solution(&self.best_solution.as_ref().unwrap(), &self.distances);
            println!("Optimal solution has Loss {}",&opt_fitness);
        }

        // TODO also plot solution
        plot_solution(best_cell.solution,&self.best_solution, &coordinates)?;
        Ok(())
    }

    /// Initialise the population for the algorithm with random solutions
    fn init_population(&self) -> Vec<Vec<Cell>> {

        let mut population = Vec::new();
        let n_nodes: usize = self.distances.shape().0;

        for _i in 0..self.individual_nb {
            let mut individual = Vec::new();
            
            for _c in 0..self.cells_nb {
                
                //Make a random solution
                let mut sol: Vec<usize> = (0..n_nodes).into_iter().collect();
                sol.shuffle(&mut rand::thread_rng());
                
                //Measure the fitness
                let cell_fit: f32 = evaluate_solution(&sol,&self.distances);
                
                let cell = Cell{
                    solution: sol,
                    fitness: cell_fit,
                    nucleosome: vec![false;n_nodes],
                    father: Option::None,
                    mother: Option::None,
                };
                individual.push(cell);
            }
            population.push(individual)
        }
        population
    }

    /// Iters through all the cells in the population 
    /// to apply the mutation mechanism.
    /// Returns the population with the mutations applied.
    fn epigen_mechanism(&self, mut population: Vec<Vec<Cell>>) -> Vec<Vec<Cell>> {

        population.par_iter_mut().for_each(|ind|{
            for cell in ind.iter_mut(){
                self.apply_mechanism(cell);
            }
        });
        population
    }
    
    /// Applies the mutations mechanisms dictated to the given cell.
    /// 
    /// # Arguments
    /// 
    /// * `cell` - Cell to apply mutations
    fn apply_mechanism(&self, cell:&mut Cell){
        for mech in self.mechanisms.iter() {
            match mech {
                Mechanisms::Imprinting(prob) => self.imprinting_mechanism(cell,prob),
                Mechanisms::Reprogramming(prob) => self.reprogramming_mechanism(cell, prob),
                Mechanisms::Position(prob) => self.position_mechanism(cell, prob),
            }
        }
    }

    /// Applies the position mutation mechanisms to the given cell.
    /// Where the nucleosome is affected with a certain probability,
    /// a _gene_ will change of place with another affected _gene_.
    ///
    /// # Arguments
    /// 
    /// * `cell` - Cell to apply mutation
    /// * `prob` - Probability to apply mutation in gene affected
    fn position_mechanism(&self, cell: &mut Cell, prob:&f32) {

        // Get the indexes for the genes that will be affected
        let mut affected_indexes: Vec<usize> = Vec::new();
        for i in 0..cell.nucleosome.len() {
            if cell.nucleosome[i] && &rand::random::<f32>() < prob{
                affected_indexes.push(i);
            }
        }

        let mut relocation = affected_indexes.clone();
        relocation.shuffle(&mut rand::thread_rng());

        let mut new_solution = cell.solution.clone();

        // Change the positions
        for i in 0..relocation.len(){
            new_solution[affected_indexes[i]] = cell.solution[relocation[i]];
        }

        cell.fitness = evaluate_solution(&new_solution, &self.distances);
        cell.solution = new_solution;
    }

    /// Applies the reprogramming mutation mechanism to the given cell.
    /// Choose some genes random _genes_ where the nucleosome is affected,
    /// and makes the crossover again. If it yields a better result
    /// the resulting solution if maintained.
    /// 
    /// # Arguments
    /// 
    /// * `cell` - Cell to apply mutation
    /// * `prob` - Probability to apply mutation in gene affected
    fn reprogramming_mechanism(&self, cell: &mut Cell, prob:&f32) {
         // If there is no father or mother the mechanism can't be done
         if cell.father == None || cell.mother == None {
            return;
        }

        // Select the changes positions to be done
        let mut changes: Vec<usize> = Vec::new();
        for i in 0..cell.nucleosome.len(){
            if cell.nucleosome[i] && &rand::random::<f32>() < prob{
                changes.push(i);
            }
        }

        // Apply changes only if they improve the fitness
        let mut best_cell = cell.clone();
        for i in changes{
            let mut new_mask = cell.nucleosome.clone();
            new_mask[i] = false;
            let new_sol = self.crossover(
                cell.father.as_ref().unwrap(),
                cell.mother.as_ref().unwrap(),
                &new_mask);
            let new_fitness = evaluate_solution(&new_sol, &self.distances);
            if new_fitness < best_cell.fitness {
                best_cell.fitness = new_fitness;
                best_cell.solution = new_sol;
                best_cell.nucleosome = new_mask;
            }
        }

        // Apply changes to the cell if improvement found
        if best_cell.fitness < cell.fitness {
            cell.fitness = best_cell.fitness;
            cell.solution = best_cell.solution;
            cell.nucleosome = best_cell.nucleosome;
        }
    }

    /// Applies the imprinting mutation mechanism to the given cell.
    /// Changes the provenience of a gene with random probabibility `prob`
    /// and where the nucleosome is affected.
    /// 
    /// # Arguments
    /// 
    /// * `cell` - Cell to apply mutation
    /// * `prob` - Probability to apply mutation in gene affected
    fn imprinting_mechanism(&self, cell:&mut Cell, prob:&f32) {
        // If there is no father or mother the mechanism can't be done
        if cell.father == None || cell.mother == None {
            return;
        }

        let father = cell.father.clone().unwrap();
        let mother = cell.mother.clone().unwrap();
        for i in 0..cell.nucleosome.len(){ 
            
            //If there is no change pass
            if cell.nucleosome[i] == false || &rand::random::<f32>() > prob{
                continue;
            }

            // From which parent the gene is?
            let mut parent_change = &mother[i];
            if cell.solution[i] != father[i]{
                parent_change = &father[i];
            }

            // Which is the value to swap and avoid replication?
            let value_replace = cell.solution[i];
            let index_replace = cell.solution
                                        .iter()
                                        .position(|x| x == parent_change)
                                        .unwrap();

            // Change the value from the other parent
            cell.solution[i] = *parent_change;
            // Swap-it of the place where it was from
            cell.solution[index_replace] = value_replace;
        }
    }
    
    /// Generates a new population of individuals from the crossover
    /// of the given population.
    /// 
    /// # Arguments
    /// 
    /// * `population` - Original population to create the new population from.
    fn nucleosome_reproduction(&self, population:Vec<Vec<Cell>>) -> Vec<Vec<Cell>>{
        let mut newpop: Vec<Vec<Cell>> = Vec::new();
        let max_iter = 2*self.individual_nb;
        let mut random_seed = rand::thread_rng();
        for _i in 0..max_iter {
            let i1 = self.roulette_selection(&population, &mut random_seed);
            let i2 = self.roulette_selection(&population, &mut random_seed);
            
            //Select best cells
            let best_cell1 = i1.iter()
            .min_by(
                |a, b|
                a.fitness.partial_cmp(&b.fitness).unwrap()
            )
            .unwrap();
            let best_cell2 = i2.iter()
            .min_by(
                |a, b|
                a.fitness.partial_cmp(&b.fitness).unwrap()
            )
            .unwrap();
            
            // Logical OR of the bestCells nucleosomes
            let new_nucleosome: Vec<bool> = best_cell1.nucleosome.iter()
                                .zip(best_cell2.nucleosome.iter())
                                .map(|(a,b)| *a || *b)
                                .collect();
            
            // Create children solutions
            let father_solution = self.crossover(
                &best_cell1.solution,
                &best_cell2.solution,
                &new_nucleosome
            );
            let mother_solution = self.crossover(
                &best_cell2.solution,
                &best_cell1.solution,
                &new_nucleosome
            );
            let new_cell_i1 = Cell{
                fitness: evaluate_solution(&father_solution,&self.distances),
                solution: father_solution.clone(),
                nucleosome: new_nucleosome.clone(),
                father: Some(father_solution.clone()),
                mother: Some(mother_solution.clone()),
            };
            let new_cell_i2 = Cell{
                fitness: evaluate_solution(&mother_solution,&self.distances),
                solution: mother_solution.clone(),
                nucleosome: new_nucleosome.clone(),
                father: Some(father_solution.clone()),
                mother: Some(mother_solution.clone()),
            };

            let i1_child:Vec<Cell> = self.remove_worst_cell(i1, new_cell_i1);
            let i2_child:Vec<Cell> = self.remove_worst_cell(i2, new_cell_i2);
            newpop.push(i1_child);
            newpop.push(i2_child);
        }
        newpop
    }

    fn crossover(&self, base_sol:&Vec<usize>, second_sol:&Vec<usize>, mask:&Vec<bool>) -> Vec<usize>{
        
        // Define the mapping for the substitution of values so that they are not repeated
        let mut mapping = HashMap::new();
        for i in 0..mask.len() {
            if mask[i] {
                mapping.insert(base_sol[i], second_sol[i]);
            }
        }

        let mut new_sol: Vec<usize> = base_sol.clone();

        for i in 0..mask.len(){
            // If the chromosome is not bent, we must replace with second solution value
            if !mask[i] {
                let mut city = second_sol[i];
                // However, if it is going to be a repeated value, we use the generated map

                while mapping.contains_key(&city) {
                    city = mapping[&city];
                }
                new_sol[i] = city;
            }
        }

        // https://www.researchgate.net/publication/226665831_Genetic_Algorithms_for_the_Travelling_Salesman_Problem_A_Review_of_Representations_and_Operators

        new_sol
    }

    fn remove_worst_cell(&self, mut individual:Vec<Cell>, new_cell: Cell) -> Vec<Cell> {
        individual.sort_by(
            |a,b|
            a.fitness.partial_cmp(&b.fitness).unwrap()
        );
        individual.remove(individual.len()-1);
        individual.push(new_cell);
        individual
    }

    fn roulette_selection(&self, population: &Vec<Vec<Cell>>, random_seed: &mut ThreadRng) -> Vec<Cell>{
        let individual_fitness: Vec<f32> = population.into_iter()
                                    .map(|ind| evaluate_individual(ind))
                                    .collect();

        //Prepare distribution criteria for random selection
        let min_fitness: f32 = individual_fitness.iter()
                                .fold(f32::INFINITY, |a, &b| a.min(b));
        let max_fitness: f32 = individual_fitness.iter()
                                .fold(f32::MIN, |a, &b| a.max(b));
        let media: f32 = ((min_fitness + max_fitness)/2.0).floor();
        let inverted: Vec<f32> = individual_fitness.iter()
                                .map(|x| -(x-media)).collect();
        let reverted: Vec<f32> = inverted.iter()
                                .map(|x| x+media).collect();
        let max_distribution: f32 = reverted.iter()
                                .fold(0.0, |a,&b| a + b);

        let pick: f32 = Uniform::from(0.0..max_distribution).sample(random_seed);
        let mut current: f32 = 0.0;
        for i in 0..population.len(){
            current += reverted[i];
            if current > pick {
                return population[i].clone()
            }
        }
        population[population.len()-1].clone()
    }

    fn nucleosome_generation(&self,mut population:Vec<Vec<Cell>>) -> Vec<Vec<Cell>>{
        
        for individual in population.iter_mut() {
            for cell in individual.iter_mut() {
                let mut nucleosome: Vec<bool> = vec![false;cell.nucleosome.len()];

                // See if there is a collapse in the nucleosome
                for k in 0..nucleosome.len(){
                    if rand::random::<f32>() < self.nucleo_prob {
                        nucleosome = self.collapse(nucleosome, k);
                    }
                }

                cell.nucleosome = nucleosome;
            }
        }

        population
    }

    fn collapse(&self, mut nucleosome: Vec<bool>, k: usize) -> Vec<bool>{

        // Sets the nucleosome to True around the collapse point
        for i in -self.nucleo_rad..self.nucleo_rad+1{
            let index: i32 = i + k as i32;
            //Checks that the collapsing point is in the range of nucleosome
            if index >=0 && index <nucleosome.len() as i32 {
                let j: usize = index as usize;
                nucleosome[j] = true;
            }
        }
        nucleosome
    }
    
}


//---- Other helpful functions ----

fn evaluate_individual(individual:&Vec<Cell>) -> f32 {
    let possible_min = individual.into_iter().map(|c| c.fitness).reduce(f32::min);
    possible_min.unwrap_or(f32::MAX)
}

pub fn evaluate_solution(solution:&Vec<usize>, distances:&DMatrix<f32>) -> f32{
    let mut fitness = 0.0;
    
    for i in 1..solution.len(){
        let origin: usize = solution[i-1];
        let destination: usize = solution[i];
        // println!("len {}", &distances.shape().0);
        // println!("origin {}, dest {}", &origin, &destination);
        fitness += distances[(origin,destination)];
    }

    fitness
}

pub fn select_best_cell(population:Vec<Vec<Cell>>) -> Cell{
     
    //Create dummy initial cell
    let mut best_cell: Cell = Cell{
        solution: Vec::new(),
        fitness: f32::MAX,
        nucleosome: Vec::new(),
        father: Option::None,
        mother: Option::None,
    };

    for individual in population{
        for cell in individual{
            if cell.fitness< best_cell.fitness{
                best_cell = cell
            }
        }
    }

    best_cell
}

pub fn plot_solution(solution: Vec<usize>, optimal: &Option<Vec<usize>>, coordinates: &Vec<Point2<f32>>) -> Result<(), Box<dyn std::error::Error>>{
    let root = BitMapBackend::new("solution.png", (2566, 1440)).into_drawing_area();
    root.fill(&WHITE)?;
    let root = root.margin(10,10,10,10);

    let max_x_coord = &coordinates
        .into_iter()
        .max_by(|x, y| partial_cmp(&x.x, &y.x).unwrap_or(Ordering::Equal))
        .unwrap().x;
    let max_y_coord = &coordinates
        .into_iter()
        .max_by(|x, y| partial_cmp(&x.y, &y.y).unwrap_or(Ordering::Equal))
        .unwrap().y;
    let min_x_coord = &coordinates
        .into_iter()
        .min_by(|x, y| partial_cmp(&x.x, &y.x).unwrap_or(Ordering::Equal))
        .unwrap().x;
    let min_y_coord = &coordinates
        .into_iter()
        .min_by(|x, y| partial_cmp(&x.y, &y.y).unwrap_or(Ordering::Equal))
        .unwrap().y;


    let mut chart = ChartBuilder::on(&root)
        .build_cartesian_2d(*min_x_coord..*max_x_coord, *min_y_coord..*max_y_coord)?;
    
    if optimal.is_some() {
        let mut best_sol: Vec<(f32, f32)> = optimal.as_ref().unwrap()
            .into_iter()
            .map(|i| (coordinates[i.clone()].x, coordinates[i.clone()].y))
            .collect();
        best_sol.push((coordinates[0].x,coordinates[0].y));

        chart.draw_series(LineSeries::new(
            best_sol.clone(),
            &BLUE,
        ))?;
        
    }

    let mut sol: Vec<(f32, f32)> = solution.into_iter().map(|i| (coordinates[i].x, coordinates[i].y)).collect();
    sol.push((coordinates[0].x,coordinates[0].y));

    chart.draw_series(LineSeries::new(
        sol.clone(),
        &RED,
    ))?;

    
    chart.draw_series(PointSeries::of_element(
        sol.clone(),
        2,
        &BLACK,
        & |c,s,st| {
            return EmptyElement::at(c)
            + Circle::new((0,0), s, st.filled())
        }
    ))?;
    
    root.present()?;
    
    Ok(())
}

//Calculates the initial matrix of distances
pub fn calculate_dist(coordinates: Vec<Point2<f32>>) -> DMatrix<f32> {
    let n = coordinates.len();
    let distance_matrix = DMatrix::<f32>::from_fn(
        n,
        n,
        |i,j| distance(&coordinates[i], &coordinates[j])
    );
    distance_matrix
}