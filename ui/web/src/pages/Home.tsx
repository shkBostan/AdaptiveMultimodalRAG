/**
 * Author: s Bostan
 * Created on: Nov, 2025
 */

import React from 'react';
import { RAGSearchBox } from '../components/RAGSearchBox';
import { ImageUpload } from '../components/ImageUpload';
import { AudioUpload } from '../components/AudioUpload';
import { ragClient } from '../api/ragClient';

export const Home: React.FC = () => {
  const [results, setResults] = React.useState<any>(null);
  const [isLoading, setIsLoading] = React.useState(false);

  const handleSearch = async (query: string) => {
    setIsLoading(true);
    try {
      const response = await ragClient.query(query);
      setResults(response);
    } catch (error) {
      console.error('Search error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleImageUpload = async (file: File) => {
    setIsLoading(true);
    try {
      const response = await ragClient.uploadImage(file);
      setResults(response);
    } catch (error) {
      console.error('Image upload error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  const handleAudioUpload = async (file: File) => {
    setIsLoading(true);
    try {
      const response = await ragClient.uploadAudio(file);
      setResults(response);
    } catch (error) {
      console.error('Audio upload error:', error);
    } finally {
      setIsLoading(false);
    }
  };

  return (
    <div className="min-h-screen bg-gray-50 py-8">
      <div className="container mx-auto px-4">
        <h1 className="text-4xl font-bold text-center mb-8">AdaptiveMultimodalRAG</h1>
        
        <div className="max-w-4xl mx-auto space-y-6">
          <RAGSearchBox onSearch={handleSearch} isLoading={isLoading} />
          
          <div className="flex gap-4 justify-center">
            <ImageUpload onUpload={handleImageUpload} isLoading={isLoading} />
            <AudioUpload onUpload={handleAudioUpload} isLoading={isLoading} />
          </div>

          {results && (
            <div className="mt-8 p-6 bg-white rounded-lg shadow">
              <h2 className="text-2xl font-semibold mb-4">Results</h2>
              <pre className="bg-gray-100 p-4 rounded overflow-auto">
                {JSON.stringify(results, null, 2)}
              </pre>
            </div>
          )}
        </div>
      </div>
    </div>
  );
};

