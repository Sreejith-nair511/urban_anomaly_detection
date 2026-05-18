"use client";
import dynamic from 'next/dynamic';
import { Spinner } from './Card';

// Dynamically import the map client component with SSR disabled
const MapClient = dynamic(() => import('./MapClient'), {
  ssr: false,
  loading: () => <Spinner />
});

export default function BangaloreMap({ onZoneClick }: { onZoneClick?: (zone: any) => void }) {
  return <MapClient onZoneClick={onZoneClick} />;
}
