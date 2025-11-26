/**
 * Author: s Bostan
 * Created on: Nov, 2025
 */

import React, { useRef } from 'react';

interface AudioUploadProps {
  onUpload: (file: File) => void;
  isLoading?: boolean;
}

export const AudioUpload: React.FC<AudioUploadProps> = ({ onUpload, isLoading = false }) => {
  const fileInputRef = useRef<HTMLInputElement>(null);

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    if (file && file.type.startsWith('audio/')) {
      onUpload(file);
    }
  };

  return (
    <div className="w-full">
      <input
        ref={fileInputRef}
        type="file"
        accept="audio/*"
        onChange={handleFileChange}
        className="hidden"
        disabled={isLoading}
      />
      <button
        onClick={() => fileInputRef.current?.click()}
        disabled={isLoading}
        className="px-4 py-2 bg-purple-500 text-white rounded-lg hover:bg-purple-600 disabled:opacity-50"
      >
        {isLoading ? 'Uploading...' : 'Upload Audio'}
      </button>
    </div>
  );
};

