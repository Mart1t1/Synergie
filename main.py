import sys

import constants
from data_generation.exporter import export
from model import model
from training import training
from training.loader import loader


def main():
    """
    argv inputs:
    -p: 1331/1414/1304/1404/1128: the session to export
    -t: train
    """

    if len(sys.argv) == 1:
        print("usage: -p <session> -t <train>")
        return

    if "-p" in sys.argv:  # export
        session_name = sys.argv[sys.argv.index("-p") + 1]
        session = constants.sessions[session_name]
        if session is None:
            raise Exception("session not found")

        export(session["path"], session["sample_time_fine_synchro"])

    if "-t" in sys.argv:  # train
        path = "data/annodated/total/"
        dataset = loader(path)
        dataset.print_stats()
        trainer = training.Trainer(dataset, model.transformer(dropout=0.3, mlp_dropout=0.1))
        trainer.train(epochs=350)


if __name__ == "__main__":
    main()
