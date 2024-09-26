import os
import pytest

skip_test = not os.path.isdir('/dls')
reason = "This test requires files from the DLS file system"

only_dls_file_system = pytest.mark.skipif(skip_test, reason=reason)
