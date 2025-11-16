import os
import sys
sys.path.append(os.path.abspath(os.path.dirname(os.path.dirname(os.path.dirname(__file__)))))
import pytest
from graph import create_graph

workflow = create_graph()
print(getattr(workflow, "config", None))