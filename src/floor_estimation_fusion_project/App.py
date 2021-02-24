from test_main import FloorEstimation
from absl import app


# noinspection PyUnusedLocal
def main(*args):
    """
    :type args: parameters given from command line
    """
    application = FloorEstimation()
    application.floor_estimation()


if __name__ == '__main__':
    app.run(main)
