import { useState, useCallback, useRef } from 'react';
import { useNavigate } from 'react-router-dom';
import { useMutation, useQuery, useQueryClient } from '@tanstack/react-query';
import { uploadDocuments, captureCamera, getAlbums, getAlbumItems, importPhotos } from '../api';
import { useToast } from '../hooks/useToast';
import CameraCapture from '../components/CameraCapture';

function AlbumBrowser({ onImport }) {
  const toast = useToast();
  const [selectedAlbumId, setSelectedAlbumId] = useState(null);
  const [selectedPhotos, setSelectedPhotos] = useState(new Set());

  const { data: albums, isLoading: albumsLoading } = useQuery({
    queryKey: ['albums'],
    queryFn: getAlbums,
  });

  const { data: albumItems, isLoading: itemsLoading } = useQuery({
    queryKey: ['albumItems', selectedAlbumId],
    queryFn: () => getAlbumItems(selectedAlbumId),
    enabled: !!selectedAlbumId,
  });

  const importMutation = useMutation({
    mutationFn: ({ photoIds, albumId }) => importPhotos(photoIds, albumId),
    onSuccess: (data) => {
      toast.success('Photos imported successfully');
      onImport(data);
    },
    onError: () => {
      toast.error('Failed to import photos.');
    },
  });

  const togglePhoto = (photoId) => {
    setSelectedPhotos((prev) => {
      const next = new Set(prev);
      if (next.has(photoId)) {
        next.delete(photoId);
      } else {
        next.add(photoId);
      }
      return next;
    });
  };

  const importSelected = () => {
    if (selectedPhotos.size === 0) return;
    importMutation.mutate({
      photoIds: Array.from(selectedPhotos),
      albumId: selectedAlbumId,
    });
  };

  const importAll = () => {
    if (!albumItems?.length) return;
    const allIds = albumItems.map((item) => item.id);
    importMutation.mutate({
      photoIds: allIds,
      albumId: selectedAlbumId,
    });
  };

  // Album grid
  if (!selectedAlbumId) {
    return (
      <div>
        <h3 className="text-sm font-semibold text-gray-700 mb-3">Google Photos Albums</h3>
        {albumsLoading ? (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {[1, 2, 3, 4, 5, 6].map((i) => (
              <div key={i} className="skeleton aspect-square rounded-lg" />
            ))}
          </div>
        ) : !albums || albums.length === 0 ? (
          <p className="text-sm text-gray-400 text-center py-8">No albums found in your Google Photos.</p>
        ) : (
          <div className="grid grid-cols-2 sm:grid-cols-3 gap-3">
            {albums.map((album) => (
              <button
                key={album.id}
                onClick={() => {
                  setSelectedAlbumId(album.id);
                  setSelectedPhotos(new Set());
                }}
                className="group text-left bg-gray-50 rounded-lg overflow-hidden border border-gray-200 hover:border-primary-300 hover:shadow-md transition-all"
              >
                <div className="aspect-square bg-gray-200 overflow-hidden">
                  {album.cover_photo_url ? (
                    <img src={album.cover_photo_url} alt="" className="w-full h-full object-cover group-hover:scale-105 transition-transform" loading="lazy" />
                  ) : (
                    <div className="w-full h-full flex items-center justify-center text-gray-400">
                      <svg className="w-8 h-8" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M4 16l4.586-4.586a2 2 0 012.828 0L16 16m-2-2l1.586-1.586a2 2 0 012.828 0L20 14m-6-6h.01M6 20h12a2 2 0 002-2V6a2 2 0 00-2-2H6a2 2 0 00-2 2v12a2 2 0 002 2z" />
                      </svg>
                    </div>
                  )}
                </div>
                <div className="p-2">
                  <p className="text-sm font-medium text-gray-900 truncate">{album.title}</p>
                  <p className="text-xs text-gray-400">{album.item_count ?? '?'} items</p>
                </div>
              </button>
            ))}
          </div>
        )}
      </div>
    );
  }

  // Album items view
  return (
    <div>
      <div className="flex items-center justify-between mb-3">
        <button
          onClick={() => {
            setSelectedAlbumId(null);
            setSelectedPhotos(new Set());
          }}
          className="flex items-center gap-1 text-sm text-gray-500 hover:text-gray-700"
        >
          <svg className="w-4 h-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
          Back to albums
        </button>
        <div className="flex items-center gap-2">
          {selectedPhotos.size > 0 && (
            <button
              onClick={importSelected}
              disabled={importMutation.isPending}
              className="px-3 py-1.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
            >
              Import {selectedPhotos.size} selected
            </button>
          )}
          <button
            onClick={importAll}
            disabled={importMutation.isPending || !albumItems?.length}
            className="px-3 py-1.5 bg-white border border-gray-300 text-gray-700 rounded-lg text-sm font-medium hover:bg-gray-50 disabled:opacity-50 transition-colors"
          >
            Import All
          </button>
        </div>
      </div>

      {itemsLoading ? (
        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
          {[1, 2, 3, 4, 5, 6, 7, 8].map((i) => (
            <div key={i} className="skeleton aspect-square rounded-lg" />
          ))}
        </div>
      ) : (
        <div className="grid grid-cols-3 sm:grid-cols-4 gap-2">
          {albumItems?.map((item) => {
            const isSelected = selectedPhotos.has(item.id);
            return (
              <button
                key={item.id}
                onClick={() => togglePhoto(item.id)}
                className={`relative aspect-square rounded-lg overflow-hidden border-2 transition-all ${
                  isSelected
                    ? 'border-primary-500 shadow-md'
                    : 'border-transparent hover:border-gray-300'
                }`}
              >
                <img
                  src={item.thumbnail_url || item.base_url}
                  alt=""
                  className="w-full h-full object-cover"
                  loading="lazy"
                />
                {isSelected && (
                  <div className="absolute inset-0 bg-primary-500/20 flex items-center justify-center">
                    <div className="w-6 h-6 bg-primary-600 rounded-full flex items-center justify-center">
                      <svg className="w-4 h-4 text-white" fill="none" stroke="currentColor" viewBox="0 0 24 24">
                        <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M5 13l4 4L19 7" />
                      </svg>
                    </div>
                  </div>
                )}
              </button>
            );
          })}
        </div>
      )}

      {importMutation.isPending && (
        <div className="mt-4 flex items-center justify-center gap-2 text-sm text-primary-600">
          <div className="w-4 h-4 border-2 border-primary-600 border-t-transparent rounded-full animate-spin" />
          Importing photos...
        </div>
      )}
    </div>
  );
}

export default function Upload() {
  const navigate = useNavigate();
  const toast = useToast();
  const queryClient = useQueryClient();
  const fileInputRef = useRef(null);
  const [showCamera, setShowCamera] = useState(false);
  const [isDragging, setIsDragging] = useState(false);
  const [activeTab, setActiveTab] = useState('upload'); // 'upload' | 'camera' | 'photos'

  const uploadMutation = useMutation({
    mutationFn: uploadDocuments,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      toast.success('Files uploaded successfully');
      if (data?.id) {
        navigate(`/documents/${data.id}`);
      } else {
        navigate('/documents');
      }
    },
    onError: () => {
      toast.error('Upload failed. Please try again.');
    },
  });

  const cameraMutation = useMutation({
    mutationFn: captureCamera,
    onSuccess: (data) => {
      queryClient.invalidateQueries({ queryKey: ['documents'] });
      setShowCamera(false);
      toast.success('Photo captured and uploaded');
      if (data?.id) {
        navigate(`/documents/${data.id}`);
      } else {
        navigate('/documents');
      }
    },
    onError: () => {
      toast.error('Failed to upload captured photo.');
    },
  });

  const handleFiles = useCallback(
    (files) => {
      if (files && files.length > 0) {
        uploadMutation.mutate(Array.from(files));
      }
    },
    [uploadMutation]
  );

  const handleDrop = useCallback(
    (e) => {
      e.preventDefault();
      setIsDragging(false);
      handleFiles(e.dataTransfer.files);
    },
    [handleFiles]
  );

  const handleDragOver = (e) => {
    e.preventDefault();
    setIsDragging(true);
  };

  const handleDragLeave = () => {
    setIsDragging(false);
  };

  return (
    <div className="max-w-3xl mx-auto px-4 py-6">
      <div className="flex items-center gap-3 mb-6">
        <button
          onClick={() => navigate('/documents')}
          className="p-1.5 text-gray-400 hover:text-gray-600 rounded-lg hover:bg-gray-100 transition-colors"
        >
          <svg className="w-5 h-5" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={2} d="M15 19l-7-7 7-7" />
          </svg>
        </button>
        <h1 className="text-xl font-bold text-gray-900">Add Documents</h1>
      </div>

      {/* Tab bar */}
      <div className="flex border-b border-gray-200 mb-6">
        {[
          { key: 'upload', label: 'Upload Files' },
          { key: 'camera', label: 'Camera' },
          { key: 'photos', label: 'Google Photos' },
        ].map((tab) => (
          <button
            key={tab.key}
            onClick={() => setActiveTab(tab.key)}
            className={`px-4 py-2.5 text-sm font-medium border-b-2 transition-colors ${
              activeTab === tab.key
                ? 'border-primary-600 text-primary-600'
                : 'border-transparent text-gray-500 hover:text-gray-700'
            }`}
          >
            {tab.label}
          </button>
        ))}
      </div>

      {/* Upload tab */}
      {activeTab === 'upload' && (
        <div
          className={`border-2 border-dashed rounded-xl p-12 text-center transition-colors ${
            isDragging ? 'drop-zone-active border-primary-400' : 'border-gray-300 hover:border-gray-400'
          }`}
          onDragOver={handleDragOver}
          onDragLeave={handleDragLeave}
          onDrop={handleDrop}
        >
          <svg className="w-12 h-12 text-gray-300 mx-auto mb-4" fill="none" stroke="currentColor" viewBox="0 0 24 24">
            <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M7 16a4 4 0 01-.88-7.903A5 5 0 1115.9 6L16 6a5 5 0 011 9.9M15 13l-3-3m0 0l-3 3m3-3v12" />
          </svg>
          <p className="text-gray-600 mb-1">
            {isDragging ? 'Drop files here' : 'Drag and drop files here'}
          </p>
          <p className="text-sm text-gray-400 mb-4">
            Supports images (JPG, PNG, HEIC) and PDFs
          </p>
          <button
            onClick={() => fileInputRef.current?.click()}
            disabled={uploadMutation.isPending}
            className="px-6 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 disabled:opacity-50 transition-colors"
          >
            {uploadMutation.isPending ? (
              <span className="flex items-center gap-2">
                <div className="w-4 h-4 border-2 border-white border-t-transparent rounded-full animate-spin" />
                Uploading...
              </span>
            ) : (
              'Choose Files'
            )}
          </button>
          <input
            ref={fileInputRef}
            type="file"
            accept="image/*,.pdf"
            multiple
            onChange={(e) => handleFiles(e.target.files)}
            className="hidden"
          />
        </div>
      )}

      {/* Camera tab */}
      {activeTab === 'camera' && (
        <div className="text-center py-8">
          <div className="w-16 h-16 bg-gray-100 rounded-2xl flex items-center justify-center mx-auto mb-4">
            <svg className="w-8 h-8 text-gray-400" fill="none" stroke="currentColor" viewBox="0 0 24 24">
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M3 9a2 2 0 012-2h.93a2 2 0 001.664-.89l.812-1.22A2 2 0 0110.07 4h3.86a2 2 0 011.664.89l.812 1.22A2 2 0 0018.07 7H19a2 2 0 012 2v9a2 2 0 01-2 2H5a2 2 0 01-2-2V9z" />
              <path strokeLinecap="round" strokeLinejoin="round" strokeWidth={1.5} d="M15 13a3 3 0 11-6 0 3 3 0 016 0z" />
            </svg>
          </div>
          <p className="text-gray-600 mb-1">Capture a photo of handwritten text</p>
          <p className="text-sm text-gray-400 mb-4">
            Use your device's camera to take a picture of a page
          </p>
          <button
            onClick={() => setShowCamera(true)}
            className="px-6 py-2.5 bg-primary-600 text-white rounded-lg text-sm font-medium hover:bg-primary-700 transition-colors"
          >
            Open Camera
          </button>
        </div>
      )}

      {/* Google Photos tab */}
      {activeTab === 'photos' && (
        <AlbumBrowser
          onImport={(data) => {
            queryClient.invalidateQueries({ queryKey: ['documents'] });
            if (data?.id) {
              navigate(`/documents/${data.id}`);
            } else {
              navigate('/documents');
            }
          }}
        />
      )}

      {/* Camera modal */}
      {showCamera && (
        <CameraCapture
          onCapture={(base64) => cameraMutation.mutate(base64)}
          onClose={() => setShowCamera(false)}
        />
      )}
    </div>
  );
}
