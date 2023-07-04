import time
import queue
import threading

RESPONSE_TIMEOUT = 10
_status_dict = {}
'''
    contains id: (status, text), with the following states:
    Solving: game is currently being solved, text describes which step of the calculation is active
    Completed: solve has been completed, result can be collected from the text
    Aborted: solve has been aborted, text describes why
    Queue: waiting for access to wolframalpha, text contains information about Queue position
    Unknown: returned if the id is not the dict
'''
_time_dict = {}
'''
    contains id: (time, event)
    where time is the last timestamp that the ids status was requested
    and event is a thread event that is set when the id is no longer requested.
    if time was more than RESPONSE_TIMEOUT seconds ago, the status is discarded and the event is set
'''
_status_queue = queue.Queue()
worker_thread = None


def get_status(id, update_time=True):
    job = ["get", False, (id, update_time)]
    _ensure_started()
    _status_queue.put(job)
    while not job[1]:
        continue
    return job[2]


def update_status(id, state, text,):
    job = ["update", False, (id, state, text)]
    _ensure_started()
    _status_queue.put(job)
    return


def add_status(id, state, text, event):
    job = ["add", False, (id, state, text, event)]
    _ensure_started()
    _status_queue.put(job)
    return


def is_alive(id):
    status = get_status(id, False)
    return status[0] not in ["Unknown", "Aborted"]


def _ensure_started():
    global worker_thread
    if worker_thread is None or not worker_thread.is_alive():
        worker_thread = threading.Thread(target=_worker, args=())
        worker_thread.start()


def _worker():
    while True:
        job = _status_queue.get()
        job_type, job_done, value = job
        if job_type == "add":
            id, state, text, event = value
            _status_dict[id] = (state, text)
            _time_dict[id] = (time.time(), event)
        elif job_type == "get":
            id, update_time = value
            if id in _status_dict:
                job[2] = (_status_dict[id][0], _status_dict[id][1])
                if update_time:
                    _time_dict[id] = (time.time(), _time_dict[id][1])
            else:
                job[2] = ("Unknown", "Request Lost")
        elif job_type == "update":
            id, state, text = value
            if id in _status_dict:
                _status_dict[id] = (state, text)
        delete_ids = []
        timestamp = time.time()
        for id in _time_dict:
            t, event = _time_dict[id]
            if timestamp - t > RESPONSE_TIMEOUT:
                delete_ids.append(id)
                event.set()
        for id in delete_ids:
            del _time_dict[id]
            del _status_dict[id]
        job[1] = True
        _status_queue.task_done()






