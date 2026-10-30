"""Bounded local TICK restore probe. Never logs credentials or authenticated URLs.

This is environment admission evidence, NOT a training launcher.
"""
import argparse
import datetime
import json
from pathlib import Path
import sys
import threading
import time
import urllib.parse
import urllib.request


def main():
    parser = argparse.ArgumentParser(__doc__)
    parser.add_argument('installation', type=Path)
    parser.add_argument('dependencies', type=Path)
    parser.add_argument('report', type=Path)
    parser.add_argument('--restore-clip', required=True)
    args = parser.parse_args()
    if args.report.exists():
        raise FileExistsError(args.report)
    sys.path.insert(0, str(args.dependencies))
    import websocket

    raw = (args.installation / 'ui/runtime-config.js').read_text(encoding='utf-8')
    config = json.loads(raw.split('=', 1)[1].strip().rstrip(';'))
    origin = config['dataServiceOrigin']
    parsed = urllib.parse.urlsplit(origin)
    if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
        raise ValueError('only loopback TICK is allowed')
    headers = {'X-Tick-Data-Service-Key': config['dataServiceApiKey'],
               'Content-Type': 'application/json'}

    def request(endpoint, payload=None):
        data = None if payload is None else json.dumps(payload).encode()
        req = urllib.request.Request(origin + '/api/' + endpoint, data=data, headers=headers)
        with urllib.request.urlopen(req, timeout=3) as response:
            return json.load(response)

    status = request('runtime/status')
    uid = 'oqIJ5rQDRrNwLPTh9H2p_W4tLof'
    if status.get('currentMapUid') != uid or status.get('effectiveGameSpeed') != 0:
        raise RuntimeError('requires paused tmrl-test')
    query = urllib.parse.urlencode({'afterSequence': 9007199254740991,
                                   'accessToken': config['dataServiceApiKey']})
    ws = websocket.create_connection('ws://' + parsed.netloc + '/events?' + query,
                                     origin='https://tick-ui.local', timeout=1)
    events = []
    failures = []
    stop = threading.Event()

    def read_events():
        while not stop.is_set():
            try:
                value = json.loads(ws.recv())
                event = value.get('runtimeEvent', {})
                if event.get('eventType') in ('vehicle-state', 'state-replay',
                                             'state-clip-catalog', 'revision-rewind'):
                    events.append(event)
                    if len(events) >= 100:
                        break
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as error:
                failures.append(type(error).__name__)
                break

    reader = threading.Thread(target=read_events, daemon=True)
    reader.start()
    outcome = {'status': 'not_requested'}
    try:
        command = request('runtime/replay-state-clip?requestedBy=ui',
                          {'runtimeClipId': args.restore_clip, 'expectedMapUid': uid})
        deadline = time.monotonic() + 8
        outcome = command
        while time.monotonic() < deadline:
            time.sleep(.25)
            commands = request('runtime/commands?limit=10')
            outcome = next((x for x in commands if x['id'] == command['id']), outcome)
            if outcome['status'] in ('succeeded', 'failed'):
                break
        time.sleep(1)
    except Exception as error:
        failures.append(type(error).__name__)
        outcome = dict(outcome, probe_error=type(error).__name__)
    finally:
        stop.set()
        reader.join(2)
        ws.close()
    result = {'utc': datetime.datetime.now(datetime.timezone.utc).isoformat(),
              'status_before': status, 'restore_command': outcome,
              'events': events, 'reader_errors': failures,
              'training_started': False, 'synchronous_step_verified': False}
    args.report.write_text(json.dumps(result, indent=2), encoding='utf-8')
    print(json.dumps({'restore_status': outcome['status'],
                      'event_types': [x['eventType'] for x in events],
                      'report': str(args.report)}))


if __name__ == '__main__':
    main()
