"""Download official v4.0 schemas, then perform offline XSD and score conservation checks."""
import json
import xml.etree.ElementTree as ET
from collections import Counter
from pathlib import Path

from baseline.download import ROOT, open_url, sha256
from .notation import validate_xml


def prepare_schema():
    folder = ROOT/'third_party/musicxml-4.0'; folder.mkdir(parents=True, exist_ok=True)
    rows = []
    for name in ['musicxml.xsd', 'xml.xsd', 'xlink.xsd']:
        url = 'https://raw.githubusercontent.com/w3c/musicxml/v4.0/schema/'+name
        path = folder/name
        if not path.exists():
            with open_url(url) as response: path.write_bytes(response.read())
        rows.append({'file': name, 'url': url, 'sha256': sha256(path)})
    (folder/'provenance.json').write_text(json.dumps(rows, indent=2), encoding='utf-8')


def audit_scores(folder):
    from lxml import etree
    schema_folder = ROOT/'third_party/musicxml-4.0'
    class Resolver(etree.Resolver):
        def resolve(self, url, pubid, context):
            name = url.rsplit('/', 1)[-1]
            if name in ['xml.xsd', 'xlink.xsd']:
                return self.resolve_filename(str(schema_folder/name), context)
    parser = etree.XMLParser(no_network=True); parser.resolvers.add(Resolver())
    schema = etree.XMLSchema(etree.parse(str(schema_folder/'musicxml.xsd'), parser))
    checked = []
    for path in sorted(Path(folder).rglob('*.musicxml')):
        schema.assertValid(etree.parse(str(path), parser)); validate_xml(path)
        source = json.loads(path.with_suffix('.json').read_text())
        tree = ET.parse(path); actual = []
        # Treble staff alone must contain exactly one untied attack for every retained source event.
        for measure in tree.findall('./part/measure'):
            cursor = 0
            for child in measure:
                if child.tag == 'backup': cursor -= int(child.findtext('duration'))
                if child.tag == 'note':
                    if child.find('pitch') is not None and child.findtext('staff') == '1' and child.find("tie[@type='stop']") is None:
                        step = child.findtext('pitch/step'); alter = int(child.findtext('pitch/alter', '0'))
                        octave = int(child.findtext('pitch/octave'))
                        pitch = {'C':0,'D':2,'E':4,'F':5,'G':7,'A':9,'B':11}[step]+alter+(octave+1)*12
                        bar = source['grid']['bars'][int(measure.get('number'))-1]
                        actual.append((pitch, round(bar['start']*12)+cursor))
                    cursor += int(child.findtext('duration'))
        expected = [(int(n['raw']['pitch']), n['start_tick']) for n in source['notes']]
        if Counter(actual) != Counter(expected): raise ValueError(f'MusicXML lost or duplicated source attacks: {path}')
        checked.append(str(path))
    if not checked: raise ValueError('No scores found to validate.')
    result = {'musicxml_version':'4.0', 'schema_valid_scores':len(checked),
              'raw_attack_pitch_tick_multisets_preserved':True, 'bar_voice_durations_consistent':True,
              'schema_sha256': {p.name:sha256(p) for p in schema_folder.glob('*.xsd')}, 'files':checked}
    (Path(folder)/'validation.json').write_text(json.dumps(result, indent=2), encoding='utf-8')
    print('Schema and attack conservation passed:', len(checked), 'scores', flush=True)
    return result


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(); parser.add_argument('folder', nargs='?', default=str(ROOT/'outputs/rhythm'))
    parser.add_argument('--prepare-schema', action='store_true'); arguments = parser.parse_args()
    if arguments.prepare_schema: prepare_schema()
    else: audit_scores(arguments.folder)
