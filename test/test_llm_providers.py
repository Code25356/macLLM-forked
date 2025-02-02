import unittest
from unittest.mock import patch, MagicMock
import os
import json
from macllm.llm_providers import OpenAIProvider, OllamaProvider

class TestOpenAIProvider(unittest.TestCase):
    def setUp(self):
        self.api_key = "test_key"
        self.provider = OpenAIProvider(self.api_key)

    def test_generate(self):
        # Setup mock response
        mock_completion = MagicMock()
        mock_completion.choices[0].message.content = "Test response"
        
        # Mock the OpenAI client's create method
        self.provider.client = MagicMock()
        self.provider.client.chat.completions.create.return_value = mock_completion

        # Test generation
        response = self.provider.generate("Test prompt")
        self.assertEqual(response, "Test response")
        
        # Verify API call
        self.provider.client.chat.completions.create.assert_called_once_with(
            model=self.provider.model,
            messages=[{"role": "user", "content": "Test prompt"}],
            temperature=self.provider.temperature
        )

    @patch('requests.post')
    def test_generate_with_image(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {
            "choices": [{"message": {"content": "Test image response"}}]
        }
        mock_post.return_value = mock_response

        # Create temp test image
        import tempfile
        with tempfile.NamedTemporaryFile(suffix='.png', delete=False) as tmp:
            tmp.write(b'test image data')
            tmp_path = tmp.name

        try:
            # Test image generation
            response = self.provider.generate_with_image("Test prompt", tmp_path)
            self.assertEqual(response, "Test image response")
            
            # Verify API call
            mock_post.assert_called_once()
            call_args = mock_post.call_args
            self.assertEqual(call_args[0][0], "https://api.openai.com/v1/chat/completions")
            self.assertEqual(call_args[1]["headers"]["Authorization"], f"Bearer {self.api_key}")
        finally:
            os.unlink(tmp_path)

class TestOllamaProvider(unittest.TestCase):
    def setUp(self):
        self.provider = OllamaProvider()

    @patch('httpx.post')
    def test_generate(self, mock_post):
        # Setup mock response
        mock_response = MagicMock()
        mock_response.status_code = 200
        mock_response.json.return_value = {"response": "Test response"}
        mock_post.return_value = mock_response

        # Test generation
        response = self.provider.generate("Test prompt")
        self.assertEqual(response, "Test response")
        
        # Verify API call
        mock_post.assert_called_once_with(
            "http://localhost:11434/api/generate",
            json={
                "model": self.provider.model,
                "prompt": "Test prompt",
                "temperature": self.provider.temperature,
                "stream": False
            }
        )

    def test_generate_with_image_not_supported(self):
        with self.assertRaises(NotImplementedError):
            self.provider.generate_with_image("Test prompt", "test.png")

if __name__ == '__main__':
    unittest.main()
