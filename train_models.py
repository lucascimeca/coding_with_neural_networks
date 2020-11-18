# The code in this script can re-train each part of the network separately. Simply uncomment the relevant part of the network to re-train and re-run the file. The new weights will be saves after a good accuracy is achieved.

from model_definitions import (
    gol_evol,
    gol_counter,
    gol_read_out
)
from mlp.data_providers import (
    GameOfLifeDataProvider,
    CounterDataProvider,
    ParityCheckDataProvider
)
import tensorflow as tf


if __name__ == "__main__":
    try:

        # FORWARD
        gol_learner_forward = gol_evol(
            steps=100000,
            exp_dir='out/forward',
            checkpoint_dir='checkpoints',
            hl_unit_no=9,
            name_scope='',
            data=GameOfLifeDataProvider(batch_size=0, shuffle_order=False, function='forward')
        )
        gol_learner_forward.start()

        # IDENTITY
        # gol_learner_identity = gol_evol(
        #     steps=100000,
        #     exp_dir='out/identity',
        #     checkpoint_dir='checkpoints',
        #     hl_unit_no=9,
        #     name_scope='',
        #     data=GameOfLifeDataProvider(batch_size=0, shuffle_order=False, function='identity')
        # )
        # gol_learner_identity.start()

        # BACKWARD
        # gol_learner_backward = gol_evol_live(
        #     steps=5000,
        #     exp_dir='out/backward',
        #     checkpoint_dir='checkpoints',
        #     hl_unit_no=100,
        #     data=GameOfLifeDataProvider(
        #         batch_size=0,              # all dataset elements in each batch
        #         shuffle_order=False,
        #         function='backward')
        # )
        # gol_learner_backward.start()

        # COUNTER
        # gol_learner_counter = gol_counter(
        #     steps=100000,
        #     exp_dir='out/counter',
        #     checkpoint_dir='checkpoints',
        #     data=CounterDataProvider(batch_size=0, shuffle_order=True)
        # )
        # gol_learner_counter.start()

        # READOUT
        # gol_learner_counter = gol_read_out(
        #     steps=100000,
        #     exp_dir='out/readout',
        #     checkpoint_dir='checkpoints',
        #     data=GameOfLifeDataProvider(batch_size=0, shuffle_order=True)
        # )
        # gol_learner_counter.start()

        # PARITY BIT CHECKER
        # gol_learner_counter = gol_evol(
        #     steps=100000,
        #     input_dim=8,
        #     out_dim=1,
        #     exp_dir='out/parity',
        #     checkpoint_dir='checkpoints',
        #     data=ParityCheckDataProvider(batch_size=0, shuffle_order=True)
        # )
        # gol_learner_counter.start()



    except Exception as e:
        print(e)
        raise Exception(e)
