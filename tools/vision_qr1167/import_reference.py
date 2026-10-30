"""Import the preserved measured line into a separate TICK collection."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time


def main():
    p = argparse.ArgumentParser(__doc__)
    p.add_argument('installation')
    p.add_argument('dependencies')
    p.add_argument('source', type=Path)
    p.add_argument('output', type=Path)
    p.add_argument('--resume-receipt', type=Path)
    a = p.parse_args()
    a.output.mkdir(exist_ok=False)
    source = a.source.read_bytes()
    digest = hashlib.sha256(source).hexdigest()
    if digest != 'a005150fc6b77163ade531407f8a0014a038206af6890edbaeb43e7c54e0e2eb':
        raise ValueError('measured reference source SHA mismatch')
    sys.path.insert(0, a.dependencies)
    from tools.vision_qr1167.tick_client import TickClient, UID
    c = TickClient(a.installation)
    report = dict(status='FAIL', source_sha256=digest, training_started=False)
    def save():
        (a.output/'result.json').write_text(json.dumps(report, indent=2), encoding='utf-8')
    try:
        report['previous_runtime'] = c.require_map()
        if report['previous_runtime']['inputExecutionEffective']:
            raise RuntimeError('existing input playback active')
        if a.resume_receipt:
            prior = json.loads(a.resume_receipt.read_text(encoding='utf-8'))
            report['collection'], report['revision'] = prior['collection'], prior['revision']
            if report['revision']['revision']['sourceSha256'] != digest:
                raise ValueError('existing revision source mismatch')
        else:
            report['collection'] = c.request('input-collections', dict(parentId=None,
                name='vision QR1167 reference 17.601', mapUid=UID, mapName='tmrl-test'), 'POST')
            save()
            report['revision'] = c.create_source(report['collection']['node']['id'], source)
        save()
        identifier = report['collection']['node']['id']
        collection = c.request('input-collections/' + identifier)
        revision = report['revision']['revision']
        report['load'] = c.request('input-revisions/' + revision['id'] +
            '/load?expectedCollectionRevision=' + str(collection['node']['revision']), {}, 'POST')
        settings = c.request('settings')
        c.request('settings', dict(expectedRevision=settings['revision'],
            activeInputCollectionId=identifier, inputExecutionEnabled=False), 'PATCH')
        deadline = time.monotonic() + 5
        while time.monotonic() < deadline:
            report['native_runtime'] = c.require_map()
            if report['native_runtime']['loadedInputRevisionId'] == revision['id']:
                break
            time.sleep(.05)
        else:
            raise TimeoutError('native did not acknowledge loaded reference')
        report['status'] = 'IMPORTED_REQUIRES_NATIVE_BASELINE'
    except Exception as error:
        report['error'] = type(error).__name__ + ': ' + str(error)
    finally:
        c.close()
        save()
    print(json.dumps(report))


if __name__ == '__main__':
    main()
