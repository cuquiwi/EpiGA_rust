use epigenetic_rust::{EpigeneticSearch,Mechanisms};
use std::{fs::File, io::{BufReader, BufRead}};
use nalgebra::{point, Point2};
use regex::Regex;
use std::error::Error;
use clap::Parser;

/// Epigenetic implementation to solve TSP
#[derive(Parser, Debug)]
#[command(author, version, about, long_about = None)]
struct Args {
   /// Name of the file containing the TSP problem
   #[arg(short, long)]
   problem: String,

   /// Max epochs to generate populations
   #[arg(short, long, default_value_t = 200)]
   epochsmax: usize,

   /// Number of individuals
   #[arg(short, long, default_value_t = 100)]
   individuals: u16,

   /// Number of cells in each individual
   #[arg(short, long, default_value_t = 10)]
   cells: u16,
}

fn main() -> Result<(), Box<dyn Error>> {

    let args = Args::parse();

    let coords = load_problem_file(&args.problem)?;

    let mut search = EpigeneticSearch::init(
        args.individuals,
        args.cells,
        0.02,
        2,
        vec![
            Mechanisms::Position(0.2),
            Mechanisms::Imprinting(0.2),
            Mechanisms::Reprogramming(0.1)
            ],
        args.epochsmax,
    );
    
    search.call(coords, None)?;

    Ok(())
}

fn load_problem_file(file_name: &str) -> Result<Vec<Point2<f32>>, Box<dyn Error>>{

    let file = File::open(file_name)?;
    let reader = BufReader::new(file);
    let mut coords: Vec<Point2<f32>> = Vec::new();

    let re = Regex::new(r"^[ \t]*\d+ \d+(\.\d+)? \d+(\.\d+)?[ \t]*$").unwrap();
    for line in reader.lines() {
        if line.is_ok(){
            let text = line?;
            if !re.is_match(text.trim()){
                continue;
            }

            let digits_text:Vec<&str> = text.trim().split(" ").collect();
            let p1: f32 = digits_text[1].parse::<f32>()?;
            let p2: f32 = digits_text[2].parse::<f32>()?;

            coords.push(point![p1,p2]);
        }
    }

    Ok(coords)
}