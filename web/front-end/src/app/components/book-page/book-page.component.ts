import { Component, OnInit } from '@angular/core';
import { BookService } from '../../book-service';
import { Book } from '../../book-service';
import { ActivatedRoute } from '@angular/router';
import { CommonModule } from '@angular/common';

@Component({
  selector: 'app-book-page',
  imports: [CommonModule],
  standalone: true,
  templateUrl: './book-page.component.html',
  styleUrl: './book-page.component.css'
})

export class BookPageComponent implements OnInit {
  book = {} as Book;
  relatedBooks:Book[];

  constructor(
    private bookService: BookService,
    private route: ActivatedRoute
  ) {
    this.relatedBooks = [];
  }

  ngOnInit() {
    this.route.paramMap.subscribe(params => {
      const id = params.get('id');
      if (id) {
        this.bookService.getBook(id).subscribe(book => {
          this.book = book;
         });
      } else {
        console.error("Book ID not found in the route.");
      }
    });
  }}
