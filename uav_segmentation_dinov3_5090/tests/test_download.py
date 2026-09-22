import io,tempfile,unittest,urllib.error,hashlib
from pathlib import Path
from unittest.mock import patch
from aic.weights import download
from aic.common import save

class Response(io.BytesIO):
    def __init__(self,data,status=200,headers=None):
        super().__init__(data); self.status=status; self.headers=headers or {'Content-Length':str(len(data))}

class DownloadTests(unittest.TestCase):
    def test_resume_and_range_ignored(self):
        url='https://example.org/weights?secret=hidden'
        for support in (True,False):
            with tempfile.TemporaryDirectory() as directory:
                target=Path(directory)/'weights.pth'; partial=target.with_suffix('.partial')
                partial.write_bytes(b'abc')
                save(partial.with_suffix('.partial.json'),{'resource':hashlib.sha256(url.split('?')[0].encode()).hexdigest(),'etag':'stable','total':6})
                response=Response(b'def',206,{'Content-Range':'bytes 3-5/6','ETag':'stable'}) if support else Response(b'abcdef')
                with patch('urllib.request.urlopen',return_value=response) as call:
                    output=download(url,target,attempts=1)
                self.assertEqual(output.read_bytes(),b'abcdef')
                self.assertEqual(call.call_args[0][0].get_header('Range'),'bytes=3-')
                self.assertNotIn('hidden',partial.with_suffix('.partial.json').read_text())

    def test_forbidden_redacts_url(self):
        with tempfile.TemporaryDirectory() as directory:
            url='https://example.org/file?secret=hidden'
            error=urllib.error.HTTPError(url,403,'Forbidden',{},None)
            with patch('urllib.request.urlopen',side_effect=error):
                with self.assertRaisesRegex(RuntimeError,'HTTP 403') as caught: download(url,Path(directory)/'w.pth')
            self.assertNotIn('hidden',str(caught.exception))

    def test_html_and_bad_range_rejected(self):
        for response in [Response(b'html',headers={'Content-Type':'text/html'}),Response(b'bad',206,{'Content-Range':'bytes 9-11/12'})]:
            with tempfile.TemporaryDirectory() as directory,patch('urllib.request.urlopen',return_value=response):
                with self.assertRaises(ValueError): download('https://example.org/file',Path(directory)/'w.pth',attempts=1)

    def test_short_response_retains_partial(self):
        with tempfile.TemporaryDirectory() as directory:
            target=Path(directory)/'w.pth'
            with patch('urllib.request.urlopen',return_value=Response(b'abc',headers={'Content-Length':'6'})):
                with self.assertRaises(RuntimeError): download('https://example.org/file',target,attempts=1)
            self.assertFalse(target.exists()); self.assertEqual(target.with_suffix('.partial').read_bytes(),b'abc')

if __name__=='__main__': unittest.main()
