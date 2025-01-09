import { Injectable } from '@angular/core';
import { HttpClient } from '@angular/common/http';
import { Observable } from 'rxjs';

export interface Book {
  name: string;
  url: string;
  id: string;
  genres: string[];
  publisher: string;
  year: number;
  ISBN: string;
  author: string;
}


@Injectable({
  providedIn: 'root'
})
export class BookService {
  private apiUrl = 'api/books'; // Replace with your actual API endpoint

  constructor(private http: HttpClient) { }

  getBooks(): Observable<Book[]> {
    return this.http.get<Book[]>(this.apiUrl);
  }

  getBook(id: string): Observable<Book> {
    const url = `${this.apiUrl}/${id}`;
    return this.http.get<Book>(url);
  }
}
