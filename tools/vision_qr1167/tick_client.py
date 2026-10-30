"""Version-specific loopback TICK client; credentials never leave memory."""
import json
from pathlib import Path
import threading
import time
import urllib.parse
import urllib.request

UID = 'oqIJ5rQDRrNwLPTh9H2p_W4tLof'


class TickClient:
    def __init__(self, installation):
        import websocket
        raw = (Path(installation) / 'ui/runtime-config.js').read_text(encoding='utf-8')
        config = json.loads(raw.split('=', 1)[1].strip().rstrip(';'))
        self.origin = config['dataServiceOrigin']
        parsed = urllib.parse.urlsplit(self.origin)
        if parsed.scheme != 'http' or parsed.hostname != '127.0.0.1':
            raise ValueError('TICK must be loopback')
        self.headers = {'X-Tick-Data-Service-Key': config['dataServiceApiKey'],
                        'Content-Type': 'application/json'}
        query = urllib.parse.urlencode({'afterSequence': 9007199254740991,
                                       'accessToken': config['dataServiceApiKey']})
        self.ws = websocket.create_connection('ws://' + parsed.netloc + '/events?' + query,
                                              origin='https://tick-ui.local', timeout=.5)
        self.events = []
        self.received_events = 0
        self.errors = []
        self.stop = threading.Event()
        self.reader = threading.Thread(target=self._read, daemon=True)
        self.reader.start()

    def _read(self):
        import websocket
        while not self.stop.is_set():
            try:
                value = json.loads(self.ws.recv())
                event = value.get('runtimeEvent')
                if event:
                    # Native payload sequences can restart after the finish/map
                    # lifecycle. This local receive cursor never restarts.
                    self.received_events += 1
                    event['_receive_index'] = self.received_events
                    self.events.append(event)
                    if len(self.events) > 10000:
                        del self.events[:5000]
            except websocket.WebSocketTimeoutException:
                continue
            except Exception as error:
                self.errors.append(type(error).__name__)
                return

    def request(self, endpoint, payload=None, method=None):
        request = urllib.request.Request(self.origin + '/api/' + endpoint,
            headers=self.headers, method=method,
            data=None if payload is None else json.dumps(payload).encode())
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    def require_map(self):
        status = self.request('runtime/status')
        if status['currentMapUid'] != UID or status['optimizationActive']:
            raise RuntimeError('requires tmrl-test without an active optimizer')
        return status

    def create_source(self, collection_id, source):
        headers = dict(self.headers, **{'Content-Type': 'text/plain; charset=utf-8'})
        request = urllib.request.Request(self.origin + '/api/input-collections/' +
            collection_id + '/revisions?origin=ui', headers=headers,
            data=source, method='POST')
        with urllib.request.urlopen(request, timeout=3) as response:
            return json.load(response)

    def command(self, name, payload, timeout=10):
        self.require_map()
        pending = self.request('runtime/' + name + '?requestedBy=ui',
                               dict(payload, expectedMapUid=UID), 'POST')
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            result = self.request('runtime/commands/' + pending['id'])
            if result['status'] == 'succeeded':
                return result
            if result['status'] in ('failed', 'cancelled', 'canceled'):
                raise RuntimeError('TICK command failed: ' + str(result.get('errorCode')))
            time.sleep(.05)
        raise TimeoutError('TICK command did not complete')

    def speed(self, value):
        self.request('runtime/game-speed', {'requestedGameSpeed': value}, 'PUT')
        deadline = time.monotonic() + 3
        while time.monotonic() < deadline:
            if self.request('runtime/status')['effectiveGameSpeed'] == value:
                return
            time.sleep(.05)
        raise TimeoutError('game speed not acknowledged')

    def close(self):
        self.stop.set()
        self.reader.join(2)
        self.ws.close()
