if __name__ == "__main__":
    
    import argparse
    
    import simulation.model.eval

    parser = argparse.ArgumentParser(description='Removing values for a parameter set of database.')
    parser.add_argument('--parameter_set', '-p', type=int, help='The parameter set which should be removed.')
    args = parser.parse_args()
    
    m = simulation.model.eval.Model()
    m._parameter_db.remove_index(args.parameter_set, force=True)
